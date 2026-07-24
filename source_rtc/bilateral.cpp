#include <array>
#include <atomic>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <iterator>
#include <memory>
#include <mutex>
#include <numbers>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <config.h>
#include "kernel.hpp"

#ifdef _MSC_VER
#   if defined (_WINDEF_) && defined(min) && defined(max)
#       undef min
#       undef max
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#endif

using namespace std::string_literals;

#define checkError(expr) do {                                                         \
    if (hipError_t result = expr; result != hipSuccess) [[unlikely]] {                \
        const char * error_str = hipGetErrorString(result);                           \
        return set_error("'"s + # expr + "' failed: " + error_str);                   \
    }                                                                                 \
} while(0)

#define checkHIPRTCError(expr) do {                                                   \
    if (hiprtcResult result = expr; result != HIPRTC_SUCCESS) [[unlikely]] {          \
        return set_error("'"s + # expr + "' failed: " + hiprtcGetErrorString(result));\
    }                                                                                 \
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
    Resource<void*, hipFree> d_src;
    Resource<void*, hipFree> d_dst;
    Resource<float *, hipHostFree> h_buffer;
    Resource<hipStream_t, hipStreamDestroy> stream;
    std::array<Resource<hipGraphExec_t, hipGraphExecDestroy>, 3> graphexecs;
};

struct BilateralData {
    VSNode * node;
    VSNode * ref_node;
    const VSVideoInfo * vi;

    // stored in graphexec
    // float sigma_spatial[3], sigma_color[3];
    // int radius[3];

    int device_id, num_streams;
    bool process[3] { true, true, true };

    int d_pitch;

    hipDevice_t device;
    hipCtx_t context; // use primary context
    ticket_semaphore semaphore;
    Resource<hipModule_t, hipModuleUnload> modules[3];
    std::vector<HIP_Resource> resources;
    std::mutex resources_lock;
};

static std::variant<hipModule_t, std::string> compile(
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    bool use_shared_memory, int block_x, int block_y, bool has_ref,
    hipDevice_t device
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    std::ostringstream kernel_source_io;
    kernel_source_io
        << std::hexfloat << std::boolalpha
        << "#define width " << width << "\n"
        << "#define height " << height << "\n"
        << "#define stride " << stride << "\n"
        << "#define sigma_spatial_scaled ((float) " << sigma_spatial_scaled << ")\n"
        << "#define sigma_color_scaled ((float)" << sigma_color_scaled << ")\n"
        << "#define radius " << radius << "\n"
        << "#define use_shared_memory " << use_shared_memory << "\n"
        << "#define BLOCK_X " << block_x << "\n"
        << "#define BLOCK_Y " << block_y << "\n"
        << "#define has_ref " << has_ref << '\n'
        << kernel_source_template;
    const std::string kernel_source = kernel_source_io.str();

    hiprtcProgram program;
    checkHIPRTCError(hiprtcCreateProgram(
        &program, kernel_source.c_str(), nullptr, 0, nullptr, nullptr));

    // Get architecture name directly for offloading
    hipDeviceProp_t props;
    checkError(hipGetDeviceProperties(&props, device));
    std::string arch_str = "--offload-arch="s + props.gcnArchName;

    const char * opts[] = {
        arch_str.c_str(),
        "-ffast-math",
        "-std=c++17"
    };

    if (hiprtcCompileProgram(program, int{std::ssize(opts)}, opts) != HIPRTC_SUCCESS) {
        size_t log_size;
        checkHIPRTCError(hiprtcGetProgramLogSize(program, &log_size));
        std::string error_message;
        error_message.resize(log_size);
        checkHIPRTCError(hiprtcGetProgramLog(program, error_message.data()));
        return set_error(error_message);
    }

    size_t code_size;
    checkHIPRTCError(hiprtcGetCodeSize(program, &code_size));
    auto image = std::make_unique<char[]>(code_size);
    checkHIPRTCError(hiprtcGetCode(program, image.get()));

    checkHIPRTCError(hiprtcDestroyProgram(&program));

    hipModule_t module_;
    checkError(hipModuleLoadData(&module_, image.get()));

    return module_;
}

static std::variant<hipGraphExec_t, std::string> get_graphexec(
    void* d_dst, void* d_src, float * h_buffer,
    int width, int height, int stride,
    int radius,
    bool use_shared_memory, int block_x, int block_y, bool has_ref,
    hipCtx_t context, hipFunction_t function
) {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    size_t pitch { stride * sizeof(float) };

    Resource<hipGraph_t, hipGraphDestroy> graph {};
    checkError(hipGraphCreate(&graph.data, 0));

    hipGraphNode_t n_HtoD;
    {
        hipMemcpy3DParms copy_params {};
        copy_params.srcPos = make_hipPos(0, 0, 0);
        copy_params.dstPos = make_hipPos(0, 0, 0);
        copy_params.srcPtr = make_hipPitchedPtr(h_buffer, pitch, width * sizeof(float), height * (1 + has_ref));
        copy_params.dstPtr = make_hipPitchedPtr(d_src, pitch, width * sizeof(float), height * (1 + has_ref));
        copy_params.kind = hipMemcpyHostToDevice;
        copy_params.extent = make_hipExtent(width * sizeof(float), height * (1 + has_ref), 1);

        checkError(hipGraphAddMemcpyNode(
            &n_HtoD, graph, nullptr, 0, &copy_params));
    }

    hipGraphNode_t n_kernel;
    {
        hipGraphNode_t dependencies[] { n_HtoD };

        void * kernelParams[] { &d_dst, &d_src };

        unsigned int sharedMemBytes = (
            use_shared_memory ?
            (1 + has_ref) * (2 * radius + block_y) * (2 * radius + block_x) * sizeof(float) :
            0
        );

        hipKernelNodeParams node_params {};
        node_params.func = (void*)function;
        node_params.gridDim = dim3(static_cast<unsigned int>((width - 1) / block_x + 1),
                                   static_cast<unsigned int>((height - 1) / block_y + 1), 1);
        node_params.blockDim = dim3(static_cast<unsigned int>(block_x),
                                    static_cast<unsigned int>(block_y), 1);
        node_params.sharedMemBytes = sharedMemBytes;
        node_params.kernelParams = kernelParams;
        node_params.extra = nullptr;

        checkError(hipGraphAddKernelNode(
            &n_kernel, graph, dependencies, std::size(dependencies), &node_params));
    }

    hipGraphNode_t n_DtoH;
    {
        hipGraphNode_t dependencies[] { n_kernel };

        hipMemcpy3DParms copy_params {};
        copy_params.srcPos = make_hipPos(0, 0, 0);
        copy_params.dstPos = make_hipPos(0, 0, 0);
        copy_params.srcPtr = make_hipPitchedPtr(d_dst, pitch, width * sizeof(float), height);
        copy_params.dstPtr = make_hipPitchedPtr(h_buffer, pitch, width * sizeof(float), height);
        copy_params.kind = hipMemcpyDeviceToHost;
        copy_params.extent = make_hipExtent(width * sizeof(float), height, 1);

        checkError(hipGraphAddMemcpyNode(
            &n_DtoH, graph, dependencies, std::size(dependencies), &copy_params));
    }

    hipGraphExec_t graphexec;
    checkError(hipGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));

    return graphexec;
}


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
            vsapi->setFilterError(("BilateralHIP: " + error_message).c_str(), frameCtx);
            if (d->ref_node) {
                vsapi->freeFrame(ref);
            }
            vsapi->freeFrame(src);
            return nullptr;
        };

        float * h_buffer = resource.h_buffer;
        hipStream_t stream = resource.stream;
        const auto & graphexecs = resource.graphexecs;

        checkError(hipCtxPushCurrent(d->context));

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
                    dst16p += s_stride;
                    h_bufferp += d_stride;
                }
            } else if (bps == 8) {
                uint8_t * dst8p = dstp;
                const float * h_bufferp = h_buffer;

                for (int y = 0; y < height; ++y) {
                    dst8p += s_stride;
                    h_bufferp += d_stride;
                }
            }
        }

        checkError(hipCtxPopCurrent(nullptr));

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

    auto device = d->device;

    hipCtxPushCurrent(d->context);

    delete d;

    hipCtxPopCurrent(nullptr);

    hipDevicePrimaryCtxRelease(device);
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
        vsapi->mapSetError(out, ("BilateralHIP: " + error_message).c_str());
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
        sigma_spatial_scaled[i] = (-0.5f / (sigma_spatial[i] * sigma_spatial[i])) * std::numbers::log2e_v<float>;
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

    // HIP related
    {
        checkError(hipInit(0));

        int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
        if (error) {
            device_id = 0;
        }

        int device_count;
        checkError(hipGetDeviceCount(&device_count));
        if (0 <= device_id && device_id < device_count) {
            checkError(hipDeviceGet(&d->device, device_id));
        } else {
            return set_error("invalid device ID (" + std::to_string(device_id) + ")");
        }
        d->device_id = device_id;

        checkError(hipDevicePrimaryCtxRetain(&d->context, d->device));
        checkError(hipCtxPushCurrent(d->context));

        d->num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
        if (error) {
            d->num_streams = 4;
        }

        bool use_shared_memory = !!vsapi->mapGetInt(in, "use_shared_memory", 0, &error);
        if (error) {
            use_shared_memory = true;
        }

        int block_x = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_x", 0, &error));
        if (error) {
            block_x = 16;
        }

        int block_y = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_y", 0, &error));
        if (error) {
            block_y = 8;
        }

        d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);

        d->resources.reserve(d->num_streams);

        int width = d->vi->width;
        int height = d->vi->height;
        int ssw = d->vi->format.subSamplingW;
        int ssh = d->vi->format.subSamplingH;

        int max_width { d->process[0] ? width : width >> ssw };
        int max_height { d->process[0] ? height : height >> ssh };

        hipFunction_t functions[3];
        for (int i = 0; i < d->num_streams; ++i) {
            Resource<void*, hipFree> d_src {};
            if (i == 0) {
                size_t d_pitch;
                checkError(hipMallocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float), (1 + has_ref) * max_height));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(hipMalloc(&d_src.data, (1 + has_ref) * max_height * d->d_pitch));
            }

            Resource<void*, hipFree> d_dst {};
            checkError(hipMalloc(&d_dst.data, max_height * d->d_pitch));

            Resource<float *, hipHostFree> h_buffer {};
            checkError(hipHostMalloc(
                reinterpret_cast<void **>(&h_buffer.data), (1 + has_ref) * max_height * d->d_pitch));

            Resource<hipStream_t, hipStreamDestroy> stream {};
            checkError(hipStreamCreateWithFlags(&stream.data, hipStreamNonBlocking));

            std::array<Resource<hipGraphExec_t, hipGraphExecDestroy>, 3> graphexecs {};
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }

                if (i == 0) {
                    const auto result = compile(
                        width, height, d->d_pitch / sizeof(float),
                        sigma_spatial_scaled[plane], sigma_color_scaled[plane], radius[plane],
                        use_shared_memory, block_x, block_y, has_ref,
                        d->device
                    );

                    if (std::holds_alternative<hipModule_t>(result)) {
                        d->modules[plane] = std::get<hipModule_t>(result);
                    } else {
                        return set_error(std::get<std::string>(result));
                    }

                    checkError(hipModuleGetFunction(
                        &functions[plane], d->modules[plane], "bilateral"));
                }

                int plane_width { plane == 0 ? width : width >> ssw };
                int plane_height { plane == 0 ? height : height >> ssh };

                const auto result = get_graphexec(
                    d_dst, d_src, h_buffer,
                    plane_width, plane_height, d->d_pitch / sizeof(float),
                    radius[plane],
                    use_shared_memory, block_x, block_y, has_ref,
                    d->context, functions[plane]
                );

                if (std::holds_alternative<hipGraphExec_t>(result)) {
                    graphexecs[plane] = std::get<hipGraphExec_t>(result);
                } else {
                    return set_error(std::get<std::string>(result));
                }
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
        "com.thefeeltrain.bilateralhip_rtc",
        "bilateralhip_rtc",
        "Bilateral filter using HIP (HIPRTC)",
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
        "block_x:int:opt;"
        "block_y:int:opt;"
        "ref:vnode:opt;",
        "clip:vnode;",
        BilateralCreate, nullptr, plugin
    );
}
