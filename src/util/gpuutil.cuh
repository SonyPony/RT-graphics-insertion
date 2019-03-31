#ifndef GPU_UTIL
#define GPU_UTIL

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define UNPACK_V2(v) v.x, v.y
#define UNPACK_V3(v) v.x, v.y, v.z
#define UNPACK_V4(v) v.x , v.y, v.z, v.w

namespace Gpu {
    namespace Utils {
        using RandState = uint32_t;

        __device__ __forceinline__ float devRand(RandState & state) {
            constexpr RandState RNG_COEFF = 4164903690U;
            state = state * RNG_COEFF + (state >> 16);
            return static_cast<float>(state) / UINT32_MAX;
        }

        inline __host__ __device__ float sum(float2 a) {
            return a.x + a.y;
        }

        inline __host__ __device__ float sum(float3 a) {
            return a.x + a.y + a.z;
        }

        void generateRandStates(RandState** d_rs, size_t count);
    }
}

#endif // !GPU_UTIL
