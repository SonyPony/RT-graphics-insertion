#ifndef GPU_UTIL
#define GPU_UTIL

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include "../common/config.h"
#include <math.h>

#define UNPACK_V2(v) v.x, v.y
#define UNPACK_V3(v) v.x, v.y, v.z
#define UNPACK_V4(v) v.x , v.y, v.z, v.w

namespace Gpu {
    namespace Utils {
        using RandState = uint32_t;

        __device__ __forceinline__ float devRand(RandState & state) {
            constexpr RandState RNG_COEFF = 4164903690U;
            state = state * RNG_COEFF + (state >> 16);
            return clamp(static_cast<float>(state) / static_cast<float>(UINT32_MAX), 0.f, 1.f);
        }

        inline __host__ __device__ float sum(float2 a) {
            return a.x + a.y;
        }

        inline __host__ __device__ float sum(float3 a) {
            return a.x + a.y + a.z;
        }

        void generateRandStates(RandState** d_rs, size_t count);

        /**
            r means radius (kernel size = 5 means r = 2)
        */
        template<typename T, typename U>
        __global__ void k_boxFilter_sep_x(T* input, U* output, int r) {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x <= r - 1 || x >= FRAME_WIDTH - r || y <= r - 1 || y >= FRAME_HEIGHT - r) {
                output[x + y * FRAME_WIDTH] = 0;
                return;
            }

            float outputValue = 0.f;
            for (int i = -r; i <= r; i++) //x
                outputValue += (float)input[x + i + y * FRAME_WIDTH];
            outputValue /= (2.f * r + 1);

            output[x + y * FRAME_WIDTH] = outputValue;
        }

        template<typename T, typename U>
        __global__ void k_boxFilter_sep_y(T* input, U* output, int r) {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x <= r - 1 || x >= FRAME_WIDTH - r || y <= r - 1 || y >= FRAME_HEIGHT - r) {
                output[x + y * FRAME_WIDTH] = 0;
                return;
            }

            float outputValue = 0.f;
            for (int j = -r; j <= r; j++) //y
                outputValue += (float)input[x + (y + j) * FRAME_WIDTH];
            outputValue /= (2.f * r + 1);

            output[x + y * FRAME_WIDTH] = outputValue;
        }

        template<typename T>
        void boxFilter(dim3 dimGrid, dim3 dimBlock, T* input, T* output, T* temp, int r) {
            k_boxFilter_sep_x<T, T> << <dimGrid, dimBlock >> > (input, temp, r);
            k_boxFilter_sep_y << <dimGrid, dimBlock >> > (temp, output, r);
        }

        template<typename T>
        void boxFilter(dim3 dimGrid, dim3 dimBlock, T* arr, T* temp, int r) {
            boxFilter<T>(dimGrid, dimBlock, arr, arr, temp, r);
        }

        void gradients(dim3 dimGrid, dim3 dimBlock, uint8_t* input, short2* temp, short2* dest);

        __host__ __device__ __inline__ uint8_t cvtRGB2GRAY(uchar4 pixel) {
            return clamp(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z, 0.f, 255.f);
        }
        

        void dualCvtRGBA2RGB(
            dim3 dimGrid, dim3 dimBlock, 
            uchar4* d_in1, uchar4* d_in2, 
            uint8_t* d_out1, uint8_t* d_out2
        );
    }
}

#endif // !GPU_UTIL
