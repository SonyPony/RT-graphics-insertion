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
            //cudaDeviceSynchronize();
            k_boxFilter_sep_y << <dimGrid, dimBlock >> > (temp, output, r);
            //cudaDeviceSynchronize();
        }

        template<typename T>
        void boxFilter(dim3 dimGrid, dim3 dimBlock, T* arr, T* temp, int r) {
            boxFilter<T>(dimGrid, dimBlock, arr, arr, temp, r);
        }

        void gradients(dim3 dimGrid, dim3 dimBlock, uint8_t* input, short2* temp, short2* dest);

        __host__ __device__ __inline__ uint8_t cvtRGB2GRAY(uchar4 pixel) {
            return clamp(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z, 0.f, 255.f);
        }

        __host__ __device__ __inline__ uchar3 cvtRGB2LAB(uchar4 pixel) {
            uchar3 result;
            const float nNormalizedR = (float)pixel.x / 255.f;//* 0.003921569F; // / 255.0F
            const float nNormalizedG = (float)pixel.y / 255.f; //* 0.003921569F;
            const float nNormalizedB = (float)pixel.z / 255.f; //* 0.003921569F;

            // to XYZ
            float nX = 0.412453f * nNormalizedR + 0.35758f  * nNormalizedG + 0.180423f * nNormalizedB;
            float nY = 0.212671f * nNormalizedR + 0.715160f  * nNormalizedG + 0.072169f * nNormalizedB;
            float nZ = 0.019334f * nNormalizedR + 0.119193f * nNormalizedG + 0.950227f * nNormalizedB;

            // stores cbrtf of nY
            float nfY = cbrtf(nY);
            float nfX = nX * 1.052128F;
            float nfZ = nZ * 0.918482F;

            float nA = cbrtf(nfX) - 16.f;
            float nB = cbrtf(nfZ) - 16.f;

            float nL = (116.f * cbrtf(nY)) - 16.f;
            /*if (nY < 0.008856f) {
                nL = 903.3 * nY;
            }*/
            nfY -= 16.f;

            nA = 500.0F * (nA - nfY);
            nB = 200.0F * (nfY - nB);

            result.x = clamp(/*nL * 2.55f*/ (nY  * 116.f - 16.f) * 2.55f, 0.f, 255.f);
            result.y = clamp(nA + 128.f, 0.f, 255.f);
            result.z = clamp(nB + 128.f, 0.f, 255.f);;

            return result;
        }

        

        void foolab(dim3 dimGrid, dim3 dimBlock, uchar4* d_frame, uint8_t* out);
    }
}

#endif // !GPU_UTIL
