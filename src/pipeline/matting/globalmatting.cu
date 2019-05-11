#include "globalmatting.cuh"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "helper_math.h"

#include "../../common/config.h"

using namespace std;


GlobalSampling::GlobalSampling(uint8_t* d_tempBuffer) {
    m_d_temp = d_tempBuffer;

    cudaMalloc(reinterpret_cast<void**>(&m_d_unknownPixels), FRAME_SIZE * sizeof(UnknownPixel));
    cudaMalloc(reinterpret_cast<void**>(&m_d_mattingSamples), FRAME_SIZE * sizeof(MattingSample));
    cudaMalloc(reinterpret_cast<void**>(&m_d_bestSamplesIndexes), FRAME_SIZE * sizeof(int));

    cudaMalloc(reinterpret_cast<void**>(&m_d_samplesCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&m_d_unknownPixelsCount), sizeof(int));

    Gpu::Utils::generateRandStates(&m_d_randStates, FRAME_SIZE);
}

GlobalSampling::~GlobalSampling() {
    cudaFree(m_d_unknownPixels);
    cudaFree(m_d_mattingSamples);
    cudaFree(m_d_bestSamplesIndexes);
    cudaFree(m_d_samplesCount);
    cudaFree(m_d_unknownPixelsCount);
}

__global__ void k_initBestSamples(
    uint8_t* trimap, MattingSample* samples, uint32_t* samplesCount, 
    int* bestSamplesIndexes, RandState* d_randState
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = FRAME_WIDTH * y + x;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1 
        || trimap[x + y * FRAME_WIDTH] != UNKNOWN) {
        return;
    }
     // random init best sample
    RandState localRandState = d_randState[id];
    const int sampleIndex = Gpu::Utils::devRand(localRandState) * ((int)(*samplesCount) - 1);
    bestSamplesIndexes[id] = min(0, sampleIndex);

    d_randState[id] = localRandState;
}

__global__ void k_initializeUnknown(
    uchar4* frame,
    uint8_t* trimap, 
    uchar4* background,
    UnknownPixel* unknownPixels,
    uint32_t* unknownPixelsCount
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int yOffset = y * FRAME_WIDTH;
    const int id = x + yOffset;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1 
        || trimap[x + y * FRAME_WIDTH] != UNKNOWN)
        return;

    const int i = atomicAdd(unknownPixelsCount, 1);
    UnknownPixel up;
    const uchar4 bgPixel = background[id];
    up.bgR = bgPixel.x;
    up.bgG = bgPixel.y;
    up.bgB = bgPixel.z;

    const uchar4 framePixel = frame[yOffset + x];
    up.frameR = framePixel.x;
    up.frameG = framePixel.y;
    up.frameB = framePixel.z;
    up.bestCost = FLT_MAX;

    up.x = x;
    up.y = y;

    unknownPixels[i] = up;
}

__global__ void k_initializeSampleSet(
    uchar4* frame,
    uint8_t* trimap,
    MattingSample* samples,
    uint32_t* samplesCount
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int yOffset = y * FRAME_WIDTH;
    const int id = x + yOffset;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1 
        || trimap[id] != FOREGROUND)
        return;
    uchar4 framePixel = frame[FRAME_WIDTH * y + x];

    // creating border between unknown area and the foreground
    if (trimap[id + 1] == UNKNOWN || trimap[id - 1] == UNKNOWN
        || trimap[id + FRAME_WIDTH] == UNKNOWN || trimap[id - FRAME_WIDTH] == UNKNOWN) {
        const int i = atomicAdd(samplesCount, 1);

        MattingSample mattingSample;
        mattingSample.x = x;
        mattingSample.y = y;
        mattingSample.R = framePixel.x;
        mattingSample.G = framePixel.y;
        mattingSample.B = framePixel.z;

        samples[i] = mattingSample;

    } // TODO rand vzorky?
}

inline __device__ float calculateAlpha(float3 I,
    float3 F, float3 B) {

    float3 numerator = (I - B) * (F - B);
    float3 denominator = (F - B) * (F - B);

    return clamp(Gpu::Utils::sum(numerator) / (1e-6f + Gpu::Utils::sum(denominator)), 0.f, 1.f);
}

inline __device__ float colorCost(float3 I,
    float3 F, float3 B, float alpha) {

    F = I - (F * alpha + B * (1.f - alpha));
    return norm3df(F.x, F.y, F.z);
}

inline __device__ float distanceColor(uint2 pixelPos, uint2 samplePos) {
    const float2 fPixelPos = make_float2(UNPACK_V2(pixelPos));
    const float2 fsamplePos = make_float2(UNPACK_V2(samplePos));

    return length(fsamplePos - fPixelPos);
}

__global__ void k_faster_sampleMatch(
    Byte* d_trimap,
    int* bestSamplesIndexes , 
    MattingSample* mattingSamples, 
    uint32_t* p_samplesCount,
    UnknownPixel* unknownPixels, 
    uint32_t* unknownPixelsCount,
    RandState* d_randState
) {

    const int id = (blockDim.x * blockIdx.x + threadIdx.x) 
        + (blockDim.y * blockIdx.y + threadIdx.y) * FRAME_WIDTH;
    const int subId = threadIdx.x * threadIdx.y;
    const int samplesCount = (*p_samplesCount);

    if (id >= (*unknownPixelsCount) || samplesCount <= 0)
        return;
    RandState localRandState = d_randState[id];
    UnknownPixel pixelInfo = unknownPixels[id];
    const int pixelId = pixelInfo.x + pixelInfo.y * FRAME_WIDTH;
    int bestSampleIndex = bestSamplesIndexes[pixelId];
    MattingSample sample = mattingSamples[bestSampleIndex];

    __shared__ int s_cost[256];

    // load pixels colors
    const float3 I = make_float3(pixelInfo.frameR, pixelInfo.frameG, pixelInfo.frameB);
    const float3 B = make_float3(pixelInfo.bgR, pixelInfo.bgG, pixelInfo.bgB);

    // init with random sample
    int sampleIndex;
    float3 F;
    float alpha;
    
    // propagation
    for (int dx = (int)pixelInfo.x - 1; dx <= (int)pixelInfo.x + 1; dx++) {
        for (int dy = (int)pixelInfo.y -1; dy <= (int)pixelInfo.y + 1; dy++) {
            int n_id = dx + dy * FRAME_WIDTH;

            if (dx < 0 || dy < 0 
                || dx > FRAME_WIDTH - 1 || dy > FRAME_HEIGHT - 1 
                || d_trimap[n_id] != UNKNOWN)
                continue;

            sampleIndex = bestSamplesIndexes[n_id];
            sample = mattingSamples[sampleIndex];

            F = make_float3(sample.R, sample.G, sample.B);

            alpha = calculateAlpha(I, F, B);
            s_cost[subId] = 10 * colorCost(I, F, B, alpha)
                + norm3df(
                    (float)sample.x - (float)pixelInfo.x,
                    (float)sample.y - (float)pixelInfo.y, 0.f
                );

            // save best sample
            if (s_cost[subId] < pixelInfo.bestCost) {
                pixelInfo.bestCost = s_cost[subId];
                pixelInfo.currentAlpha = alpha * 255;
                bestSampleIndex = sampleIndex;
            }
        }
    }

    // random walk
    for (float BK = 1.f; BK * samplesCount > 1.f; BK *= 0.5f) {
        sampleIndex = bestSampleIndex 
            + BK * (Gpu::Utils::devRand(localRandState) * 2.f - 1.f) * (samplesCount - 1);
        
        if (sampleIndex < 0 || sampleIndex >= samplesCount)
            continue;
        sample = mattingSamples[sampleIndex];

        F = make_float3(sample.R, sample.G, sample.B);
       
        alpha = calculateAlpha(I, F, B);
        s_cost[subId] = 10*colorCost(I, F, B, alpha)
            + norm3df(
                (float)sample.x - (float)pixelInfo.x, 
                (float)sample.y - (float)pixelInfo.y, 0.f
            );
        
        // save best sample
        if (s_cost[subId] < pixelInfo.bestCost) {
            pixelInfo.bestCost = s_cost[subId];
            pixelInfo.currentAlpha = alpha * 255;
            bestSampleIndex = sampleIndex;
        }
    }

    unknownPixels[id] = pixelInfo;
    d_randState[id] = localRandState;
    bestSamplesIndexes[pixelInfo.x + pixelInfo.y * FRAME_WIDTH] = bestSampleIndex;
}

__global__ void k_renderMatting(
    uint8_t* trimap, 
    UnknownPixel* unknownPixels, 
    uint32_t* unknownPixelsCount, 
    uint8_t* alphaMask
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = y * FRAME_WIDTH + x;

    const uint8_t trimapVal = trimap[id];
    if (trimapVal != UNKNOWN)
        alphaMask[id] = trimapVal;

    if (id >= (*unknownPixelsCount))
        return;

    const auto pixelInfo = unknownPixels[id];
    const int pixelId = pixelInfo.x + pixelInfo.y * FRAME_WIDTH;
    alphaMask[pixelId] = pixelInfo.currentAlpha;
}

void GlobalSampling::matting(uchar4 * d_frame, Byte * d_trimap, uchar4* d_background, Byte * d_output)
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    cudaMemset(m_d_samplesCount, 0, sizeof(uint32_t));
    cudaMemset(m_d_unknownPixelsCount, 0, sizeof(uint32_t));

    // init sample and unkown pixels structures
    k_initializeSampleSet << <dimGrid, dimBlock >> > (
        d_frame, d_trimap, m_d_mattingSamples, m_d_samplesCount
    );

    k_initializeUnknown << <dimGrid, dimBlock >> > (
        d_frame, d_trimap, d_background, m_d_unknownPixels, m_d_unknownPixelsCount
    );

    // init trimap with random samples
    k_initBestSamples << <dimGrid, dimBlock >> > (
        d_trimap, m_d_mattingSamples, m_d_samplesCount, m_d_bestSamplesIndexes, m_d_randStates
    );

    // sample patch
    for (int i = 0; i < 10; i++) {
        k_faster_sampleMatch << <dimGrid, dimBlock >> > (
            d_trimap, m_d_bestSamplesIndexes, m_d_mattingSamples, m_d_samplesCount,
            m_d_unknownPixels, m_d_unknownPixelsCount, m_d_randStates
        );
    }

    // render alpha mask
    cudaMemset(m_d_temp, 0, FRAME_SIZE);
    k_renderMatting << <dimGrid, dimBlock >> > (
        d_trimap, m_d_unknownPixels, m_d_unknownPixelsCount, m_d_temp
    );

    // guided filter
    m_guidedFilter.filter(d_frame, m_d_temp, d_output);
}