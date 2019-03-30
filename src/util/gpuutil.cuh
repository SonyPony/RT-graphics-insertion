#ifndef GPU_UTIL
#define GPU_UTIL

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Gpu {
    namespace Utils {
        using RandState = uint32_t;

        __device__ __forceinline__ float devRand(RandState & state) {
            constexpr RandState RNG_COEFF = 4164903690U;
            state = state * RNG_COEFF + (state >> 16);
            return static_cast<float>(state) / UINT32_MAX;
        }
    }
}

#endif // !GPU_UTIL
