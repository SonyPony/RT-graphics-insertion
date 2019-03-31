#include "globalmatting.cuh"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "helper_math.h"

using namespace std;

Gpu::GlobalSampling::GlobalSampling(int width, int height)
    : m_width{ width }, m_height{ height }, m_size{ width * height } {
    auto randState = new RandState[m_size];
}

__global__ void k_initializeSampleSet(
    uint8_t* trimap,
    uint8_t* dest,
    int width,
    int height,
    uint16_t* fgBorderX,
    uint16_t* fgBorderY,
    uint32_t* fgBorderCounter,

    uint16_t* unknownX,
    uint16_t* unknownY,
    uint32_t* unkownCounter
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int yOffset = y * width;

    constexpr uint8_t GRAY = 128;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || trimap[x + y * width] != 255)
        return;

    // creating border between unknown area and the foreground
    if (trimap[x + yOffset + 1] == GRAY || trimap[x + yOffset - 1] == GRAY
        || trimap[x + yOffset + width] == GRAY || trimap[x + yOffset - width] == GRAY) {
        int i = atomicAdd(fgBorderCounter, 1);
        fgBorderX[i] = x;
        fgBorderY[i] = y;
        dest[x + y * width] = 180;

    } // TODO rand vzorky?

    if (trimap[x + yOffset] == GRAY) {
        int i = atomicAdd(unkownCounter, 1);
        unknownX[i] = x;
        unknownY[i] = y;
    }

}

inline __device__ float calculateAlpha(uchar3 pixelColor,
    uchar3 sampleColor, uchar3 backgoundColor) {
    const float3 fPixelColor = make_float3(UNPACK_V3(pixelColor));
    const float3 fSampleColor = make_float3(UNPACK_V3(sampleColor));
    const float3 fBackgroundColor = make_float3(UNPACK_V3(backgoundColor));

    float3 numerator = (fPixelColor - fBackgroundColor) * (fSampleColor - fBackgroundColor);
    float3 denominator = fPixelColor - fBackgroundColor;
    denominator *= denominator;

    return Gpu::Utils::sum(numerator) / (1e-6f + Gpu::Utils::sum(denominator));
}

inline __device__ float colorCost(uchar3 pixelColor,
    uchar3 sampleColor, uchar3 backgoundColor, float alpha) {
    const float3 fPixelColor = make_float3(UNPACK_V3(pixelColor));
    const float3 fSampleColor = make_float3(UNPACK_V3(sampleColor));
    const float3 fBackgroundColor = make_float3(UNPACK_V3(backgoundColor));

    return length(fPixelColor - (fSampleColor * alpha + fBackgroundColor * (1.f - alpha)));
}

inline __device__ float distanceColor(uint2 pixelPos, uint2 samplePos) {
    const float2 fPixelPos = make_float2(UNPACK_V2(pixelPos));
    const float2 fsamplePos = make_float2(UNPACK_V2(samplePos));

    return length(fsamplePos - fPixelPos);
}

__global__ void k_sampleMatch(Byte* trimap, Byte* frame, Byte* background) {
    // propagation

    // random walk
}

void Gpu::GlobalSampling::matting(Byte * d_image, Byte * d_trimap, Byte * d_output)
{
    // MATTING
    uint16_t* d_fgBorderX = nullptr;
    uint16_t* d_fgBorderY = nullptr;
    uint32_t* fgBorderPointsCount = nullptr;

    uint16_t* d_unknownX = nullptr;
    uint16_t* d_unknownY = nullptr;
    uint32_t* d_unknownCount = nullptr;
    float* d_dists = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&fgBorderPointsCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_fgBorderX), m_size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_fgBorderY), m_size * sizeof(uint16_t));
    cudaMemset(reinterpret_cast<void*>(fgBorderPointsCount), 0, sizeof(uint16_t));

    cudaMalloc(reinterpret_cast<void**>(&d_unknownCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownX), m_size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownY), m_size * sizeof(uint16_t));
    cudaMemset(reinterpret_cast<void*>(d_unknownCount), 0, sizeof(uint16_t));


    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    k_initializeSampleSet<<<dimGrid, dimBlock>>> (
        d_trimap, d_output, 1280, 720, d_fgBorderX, d_fgBorderY, fgBorderPointsCount,
        d_unknownX, d_unknownY, d_unknownCount
    );

    //matting res trimap border fg = 7251 without rand
    uint32_t count = 0, countBorder;
    cudaMemcpy(&count, d_unknownCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&countBorder, fgBorderPointsCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "Unknown count " << countBorder << std::endl;
}