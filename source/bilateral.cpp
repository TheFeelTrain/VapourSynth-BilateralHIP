#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <numbers>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <config.h>

using namespace std::string_literals;

extern hipGraphExec_t get_graphexec(
    float * d_dst, float * d_src, float * h_buffer,
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    bool use_shared_memory, bool has_ref);

#define checkError(expr) do {                                                               \
    hipError_t __err = expr;                                                                \
    if (__err != hipSuccess) {                                                              \
        return set_error("'"s + # expr + "' failed: " + hipGetErrorString(__err));          \
    }                                                                                       \
} while(0)

struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

template <typename T, auto deleter>
    requires
        std::default_initializable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T>
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;

    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}

    [[nodiscard]] constexpr Resource(Resource&& other) noexcept
        : data(std::exchange(other.data, T{}))
    { }

    constexpr Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

struct HIP_Resource {
    Resource<float *, hipFree> d_src;
    Resource<float *, hipFree> d_dst;
    Resource<float *, hipHostFree> h_buffer;
    Resource<hipStream_t, hipStreamDestroy> stream;
    std::array<Resource<hipGraphExec_t, hipGraphExecDestroy>, 3> graphexecs;
};

struct BilateralData {
    VSNode * node;
    VSNode * ref_node;
    const VSVideoInfo * vi;

    int device_id, num_streams;
    bool process[3] { true, true, true };

    int d_pitch;
    ticket_semaphore semaphore;
    std::vector<HIP_Resource> resources;
    std::mutex resources_lock;
};

static const VSFrame *VS_CC BilateralGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->ref_node) {
            vsapi->requestFrameFilter(n, d->ref_node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VSFrame * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame * ref = nullptr;
        if (d->ref_node) {
            ref = vsapi->getFrameFilter(n, d->ref_node, frameCtx);
        }

        const int pl[] = { 0, 1, 2 };
        const VSFrame * fr[] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        VSFrame * dst = vsapi->newVideoFrame2(
            &d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        d->semaphore.acquire();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        auto set_error = [&](const std::string & error_message) {
            d->resources_lock.lock();
            d->resources.push_back(std::move(resource));
            d->resources_lock.unlock();
            d->semaphore.release();
            vsapi->setFilterError(("BilateralGPU: " + error_message).c_str(), frameCtx);
            if (d->ref_node) {
                vsapi->freeFrame(ref);
            }
            vsapi->freeFrame(src);
            return nullptr;
        };

        float * h_buffer = resource.h_buffer;
        hipStream_t stream = resource.stream;
        const auto & graphexecs = resource.graphexecs;

        for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
            if (!d->process[plane]) {
                continue;
            }

            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            int s_pitch = vsapi->getStride(src, plane);
            int bps = d->vi->format.bitsPerSample;
            int s_stride = s_pitch / (bps / 8);
            int width_bytes = width * sizeof(float);
            auto srcp = vsapi->getReadPtr(src, plane);
            int d_pitch = d->d_pitch;
            int d_stride = d_pitch / sizeof(float);

            const uint8_t * refp = nullptr;
            if (d->ref_node) {
                refp = vsapi->getReadPtr(ref, plane);
            }

            if (bps == 32) {
                vsh::bitblt(h_buffer, d_pitch, srcp, s_pitch, width_bytes, height);
                if (d->ref_node) {
                    vsh::bitblt(&h_buffer[s_stride * height], d_pitch, refp, s_pitch, width_bytes, height);
                }
            } else if (bps == 16) {
                float * h_bufferp = h_buffer;

                const auto load = [width, height, &h_bufferp, s_stride, d_stride](const uint16_t * srcp) {
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            h_bufferp[x] = static_cast<float>(srcp[x]) / 65535.0f;
                        }
                        h_bufferp += d_stride;
                        srcp += s_stride;
                    }
                };

                load(reinterpret_cast<const uint16_t *>(srcp));
                if (d->ref_node) {
                    load(reinterpret_cast<const uint16_t *>(refp));
                }
            } else if (bps == 8) {
                float * h_bufferp = h_buffer;

                const auto load = [width, height, &h_bufferp, s_stride, d_stride](const uint8_t * srcp) {
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            h_bufferp[x] = static_cast<float>(srcp[x]) / 255.f;
                        }
                        h_bufferp += d_stride;
                        srcp += s_stride;
                    }
                };

                load(srcp);
                if (d->ref_node) {
                    load(reinterpret_cast<const uint8_t *>(refp));
                }
            }

            checkError(hipGraphLaunch(graphexecs[plane], stream));
            checkError(hipStreamSynchronize(stream));

            auto dstp = vsapi->getWritePtr(dst, plane);

            if (bps == 32) {
                vsh::bitblt(dstp, s_pitch, h_buffer, d_pitch, width_bytes, height);
            } else if (bps == 16) {
                uint16_t * dst16p = reinterpret_cast<uint16_t *>(dstp);
                const float * h_bufferp = h_buffer;

                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        float dstf = h_bufferp[x] * 65535.0f;
                        dst16p[x] = static_cast<uint16_t>(std::roundf(dstf));
                    }
                    dst16p += s_stride;
                    h_bufferp += d_stride;
                }
            } else if (bps == 8) {
                uint8_t * dst8p = dstp;
                const float * h_bufferp = h_buffer;

                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        float dstf = h_bufferp[x] * 255.0f;
                        dst8p[x] = static_cast<uint8_t>(std::roundf(dstf));
                    }
                    dst8p += s_stride;
                    h_bufferp += d_stride;
                }
            }
        }

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        if (d->ref_node) {
            vsapi->freeFrame(ref);
        }
        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}

static void VS_CC BilateralFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    vsapi->freeNode(d->node);

    hipSetDevice(d->device_id);

    delete d;
}

static void VS_CC BilateralCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BilateralData>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
    bool has_ref = d->ref_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("BilateralGPU: " + error_message).c_str());
        if (has_ref) {
            vsapi->freeNode(d->ref_node);
        }
        vsapi->freeNode(d->node);
    };

    if (auto [bps, sample] = std::pair{
            d->vi->format.bitsPerSample,
            d->vi->format.sampleType
        };
        !vsh::isConstantVideoFormat(d->vi) ||
        (sample == stInteger && (bps != 8 && bps != 16)) ||
        (sample == stFloat && bps != 32)
    ) {

        return set_error("only constant format 8/16bit int or 32bit float input supported");
    }

    const auto ref_vi = vsapi->getVideoInfo(d->ref_node);
    if (d->ref_node && (!vsh::isSameVideoInfo(d->vi, ref_vi) || d->vi->numFrames != ref_vi->numFrames)) {
        return set_error("\"ref\" must be of the same format as \"clip\"");
    }

    std::array<float, 3> sigma_spatial;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial[i] = static_cast<float>(
            vsapi->mapGetFloat(in, "sigma_spatial", i, &error));

        if (error) {
            if (i == 0) {
                sigma_spatial[i] = 3.0f;
            } else if (i == 1) {
                auto subH = d->vi->format.subSamplingH;
                auto subW = d->vi->format.subSamplingW;
                sigma_spatial[i] = static_cast<float>(
                    sigma_spatial[0] / std::sqrt((1 << subH) * (1 << subW)));
            } else {
                sigma_spatial[i] = sigma_spatial[i - 1];
            }
        } else if (sigma_spatial[i] < 0.f) {
            return set_error("\"sigma_spatial\" must be non-negative");
        }

        if (sigma_spatial[i] < FLT_EPSILON) {
            d->process[i] = false;
        }
    }

    std::array<float, 3> sigma_spatial_scaled;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial_scaled[i] = -0.5f / (sigma_spatial[i] * sigma_spatial[i]) * std::numbers::log2e_v<float>;
    }

    std::array<float, 3> sigma_color;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        sigma_color[i] = static_cast<float>(
            vsapi->mapGetFloat(in, "sigma_color", i, &error));

        if (error) {
            if (i == 0) {
                sigma_color[i] = 0.02f;
            } else {
                sigma_color[i] = sigma_color[i - 1];
            }
        } else if (sigma_color[i] < 0.f) {
            return set_error("\"sigma_color\" must be non-negative");
        }
    }

    std::array<float, 3> sigma_color_scaled;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            sigma_color_scaled[i] = (-0.5f / (sigma_color[i] * sigma_color[i])) * std::numbers::log2e_v<float>;
        }
    }

    std::array<int, 3> radius;
    for (int i = 0; i < std::ssize(radius); ++i) {
        radius[i] = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", i, &error));

        if (error) {
            radius[i] = std::max(1, static_cast<int>(std::roundf(sigma_spatial[i] * 3.f)));
        } else if (radius[i] <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    int device_count;
    checkError(hipGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(hipSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    d->num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 4;
    }

    bool use_shared_memory = !!vsapi->mapGetInt(in, "use_shared_memory", 0, &error);
    if (error) {
        use_shared_memory = true;
    }

    {
        d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);

        d->resources.reserve(d->num_streams);

        int width = d->vi->width;
        int height = d->vi->height;
        int ssw = d->vi->format.subSamplingW;
        int ssh = d->vi->format.subSamplingH;

        int max_width { d->process[0] ? width : width >> ssw };
        int max_height { d->process[0] ? height : height >> ssh };

        for (int i = 0; i < d->num_streams; ++i) {
            Resource<float *, hipFree> d_src {};
            if (i == 0) {
                size_t d_pitch;
                checkError(hipMallocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float), (1 + has_ref) * max_height));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(hipMalloc(&d_src.data, (1 + has_ref) * max_height * d->d_pitch));
            }

            Resource<float *, hipFree> d_dst {};
            checkError(hipMalloc(&d_dst.data, max_height * d->d_pitch));

            Resource<float *, hipHostFree> h_buffer {};
            checkError(hipHostMalloc(&h_buffer.data, (1 + has_ref) * max_height * d->d_pitch));

            Resource<hipStream_t, hipStreamDestroy> stream {};
            checkError(hipStreamCreateWithFlags(&stream.data, hipStreamNonBlocking));

            std::array<Resource<hipGraphExec_t, hipGraphExecDestroy>, 3> graphexecs {};
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }

                int plane_width { plane == 0 ? width : width >> ssw };
                int plane_height { plane == 0 ? height : height >> ssh };

                graphexecs[plane] = get_graphexec(
                    d_dst, d_src, h_buffer,
                    plane_width, plane_height, d->d_pitch / sizeof(float),
                    sigma_spatial_scaled[plane], sigma_color_scaled[plane], radius[plane],
                    use_shared_memory, has_ref
                );
            }

            d->resources.push_back(HIP_Resource{
                .d_src = std::move(d_src),
                .d_dst = std::move(d_dst),
                .h_buffer = std::move(h_buffer),
                .stream = std::move(stream),
                .graphexecs = std::move(graphexecs)
            });
        }
    }

    VSFilterDependency deps[2] = {{d->node, rpStrictSpatial}};
    int num_deps = 1;
    if (has_ref) {
        deps[1].source = d->ref_node;
        deps[1].requestPattern = rpStrictSpatial;
        num_deps = 2;
    }

    vsapi->createVideoFilter(
        out, "Bilateral", d->vi,
        BilateralGetFrame, BilateralFree,
        fmParallel, deps, num_deps, d.release(), core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        "com.thefeeltrain.bilateralhip",
        "bilateralhip",
        "Bilateral filter using HIP",
        VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION, 0, plugin
    );

    vspapi->registerFunction(
        "Bilateral",
        "clip:vnode;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "use_shared_memory:int:opt;"
        "ref:vnode:opt;",
        "clip:vnode;",
        BilateralCreate, nullptr, plugin
    );
}
