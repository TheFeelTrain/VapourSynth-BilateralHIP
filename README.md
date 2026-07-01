# VapourSynth-BilateralHIP
Copyright© 2021 WolframRhodium<br>
Copyright© 2026 TheFeelTrain

Bilateral filter in HIP for VapourSynth. 

## Description
[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images.

The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

Special thanks to [Kice](https://github.com/kice) for doing most of the work in previous implementation.

## Requirements

- GPU(s) supported by HIP/ROCm

## Supported Formats

sample type: 8-16 bit integer or 32 bit float Gray/YUV/RGB input

## Usage

```python
core.bilateralhip.Bilateral(clip clip, float[] sigma_spatial=3.0, float[] sigma_color=0.02, int[] radius=0, int device_id=0, int num_streams=4, bool use_shared_memory=True)
```

- clip:
    The input clip.

- sigma_spatial: (Default: 3.0)
    Filter sigma in the coordinate space.
	Use an array to assign it for each plane. If "sigma_spatial" for the second plane is not specified, it will be set according to the sigma_spatial of first plane and sub-sampling.

- sigma_color: (Default: 0.02)
    Filter sigma in the color space.
	Use an array to assign it for each plane, otherwise the same sigma_color is used for all the planes.
	It will be normalized internally, so that for clips with different bit depths, the same values get similar results.

- radius: (Default: 0)
    Kernel window size. 0 = automatic calculatation based on "sigma_spatial".

- device_id: (Default: 0)
    HIP device ID.

- num_streams: (Default: 4)
    Number of HIP streams, enables concurrent kernel execution and data transfer.

- use_shared_memory: (Default: True)
    Use on-chip memory to reduce bandwidth requirements on memory operations.

## Manual Compilation
```bash
CMAKE_PREFIX_PATH=/opt/rocm \
CXX=/opt/rocm/llvm/bin/clang++ \
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```
