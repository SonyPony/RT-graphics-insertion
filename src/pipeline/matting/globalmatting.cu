#include "globalmatting.cuh"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "helper_math.h"

#define GRAY 128
using namespace std;

__constant__ int c_frameSize[2];
#define FRAME_WIDTH  1280
#define FRAME_HEIGHT 720

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> d_text_frame;

Gpu::GlobalSampling::GlobalSampling(int width, int height)
    : m_width{ width }, m_height{ height }, m_size{ width * height } {
    Gpu::Utils::generateRandStates(&m_d_randStates, m_size);
}

__global__ void k_initBestSamples(MattingSample* samples, int samplesCount, uint8_t* trimap, MattingSample** bestSamples,
    int* bestSamplesIndexes, RandState* d_randState, int width, int height) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = width * y + x;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || trimap[x + y * width] != 128) {
        bestSamples[id] = nullptr;
        return;
    }

    RandState localRandState = d_randState[id];
    int sampleIndex = Gpu::Utils::devRand(localRandState) * (samplesCount - 1);
    bestSamplesIndexes[id] = sampleIndex;
    bestSamples[id] = &samples[sampleIndex];

    d_randState[id] = localRandState;
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
    uint32_t* unkownCounter,
    ushort2* fgBoundaryPos,

    UnknownPixel* unknownPixels,
    int* unknownPixelsCount,
    uchar4* frame,
    uint8_t* background,
    MattingSample* samples
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int yOffset = y * width;
    int i;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || trimap[x + y * width] != 255)
        return;

    i = atomicAdd(unknownPixelsCount, 1);
    UnknownPixel up;
    up.bgR = background[(yOffset + x)];
    up.bgG = background[(yOffset + x)  + (width * height)];
    up.bgB = background[(yOffset + x) + 2 * (width * height)];

    uchar4 framePixel = frame[yOffset + x];
    up.frameR = framePixel.x;
    up.frameG = framePixel.y;
    up.frameB = framePixel.z;
    up.bestCost = FLT_MAX;

    up.x = x;
    up.y = y;

    unknownPixels[i] = up;

    // creating border between unknown area and the foreground
    if (trimap[x + yOffset + 1] == GRAY || trimap[x + yOffset - 1] == GRAY
        || trimap[x + yOffset + width] == GRAY || trimap[x + yOffset - width] == GRAY) {
        i = atomicAdd(fgBorderCounter, 1);
        fgBorderX[i] = x;
        fgBorderY[i] = y;
        fgBoundaryPos[i] = make_ushort2(x, y);
        dest[x + y * width] = 180;

        MattingSample mattingSample;
        mattingSample.x = x;
        mattingSample.y = y;
        mattingSample.R = framePixel.x;
        mattingSample.G = framePixel.y;
        mattingSample.B = framePixel.z;
        //mattingSample.index = i;
        

        samples[i] = mattingSample;

    } // TODO rand vzorky?

    if (trimap[x + yOffset] == GRAY) {
        i = atomicAdd(unkownCounter, 1);
        unknownX[i] = x;
        unknownY[i] = y;
    }

}

inline __device__ float calculateAlpha(float3 pixelColor,
    float3 sampleColor, float3 backgroundColor) {

    float3 numerator = (pixelColor - backgroundColor) * (sampleColor - backgroundColor);
    float3 denominator = pixelColor - backgroundColor;
    denominator *= denominator;

    return Gpu::Utils::sum(numerator) / (1e-6f + Gpu::Utils::sum(denominator));
}

inline __device__ float colorCost(float3 pixelColor,
    float3 sampleColor, float3 backgroundColor, float alpha) {

    return length(pixelColor - (sampleColor * alpha + backgroundColor * (1.f - alpha)));
}

inline __device__ float faster_colorCost(float3 pixelColor,
    float3 sampleColor, float3 backgroundColor, float alpha) {

    sampleColor = pixelColor - (sampleColor * alpha + backgroundColor * (1.f - alpha));
    return norm3df(sampleColor.x, sampleColor.y, sampleColor.z);
}

inline __device__ float distanceColor(uint2 pixelPos, uint2 samplePos) {
    const float2 fPixelPos = make_float2(UNPACK_V2(pixelPos));
    const float2 fsamplePos = make_float2(UNPACK_V2(samplePos));

    return length(fsamplePos - fPixelPos);
}

__global__ void k_sampleMatch(Byte* d_trimap, uchar4* d_frame, Byte* d_background, Byte* d_dest,
    RandState* d_randState, ushort2* fgBoundaryPos, uint16_t fgBorderCount,
    int width, int height) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * width;
    const int subId = threadIdx.x * threadIdx.y;

    RandState localRandState = d_randState[id];

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || d_trimap[id] != GRAY)
        return;

    __shared__ uchar4 s_sampleColors[256];
    __shared__ int s_cost[256];
    __shared__ int s_bestCost[256];
    __shared__ ushort2 s_fgBoundaryPos[256];

    // load pixels colors
    uchar4 pixelColor = d_frame[id];
    float3 reducedPixelColor = make_float3(pixelColor.x, pixelColor.y, pixelColor.z);
    float rawBgColor[3];

    for (int channel = 0; channel < 3; channel++)
        rawBgColor[channel] = d_background[id + channel * (width * height)];
    float3 bgColor = make_float3(rawBgColor[0], rawBgColor[1], rawBgColor[2]);

    // init with random sample
    int sampleIndex = Gpu::Utils::devRand(localRandState) * (fgBorderCount - 1);
    s_fgBoundaryPos[subId] = fgBoundaryPos[sampleIndex];

    s_sampleColors[subId] = d_frame[s_fgBoundaryPos[subId].y * width + s_fgBoundaryPos[subId].x];
    float3 reducedSampleColor = make_float3(UNPACK_V3(s_sampleColors[subId]));

    float alpha = calculateAlpha(reducedPixelColor, reducedSampleColor, bgColor);
    s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
        + sqrtf((s_fgBoundaryPos[subId].x - x) * (s_fgBoundaryPos[subId].x - x) + (s_fgBoundaryPos[subId].y - y) * (s_fgBoundaryPos[subId].y - y));
    s_bestCost[subId] = s_cost[subId];
    uint2 bestPos = make_uint2(s_fgBoundaryPos[subId].x, s_fgBoundaryPos[subId].y);
    // propagation

    for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
            int sampleIndex = x + dx + (y + dy) * width;

            if (d_trimap[sampleIndex] == GRAY) {
                s_fgBoundaryPos[subId] = fgBoundaryPos[sampleIndex];

                s_sampleColors[subId] = s_sampleColors[subId] = d_frame[s_fgBoundaryPos[subId].y * width + s_fgBoundaryPos[subId].x];
                reducedSampleColor = make_float3(UNPACK_V3(s_sampleColors[subId]));

                alpha = calculateAlpha(reducedPixelColor, reducedSampleColor, bgColor);
                s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
                    + sqrtf((s_fgBoundaryPos[subId].x - x) * (s_fgBoundaryPos[subId].x - x) + (s_fgBoundaryPos[subId].y - y) * (s_fgBoundaryPos[subId].y - y));
                if (s_cost[subId] < s_bestCost[subId]) {
                    s_bestCost[subId] = s_cost[subId];
                    bestPos = make_uint2(s_fgBoundaryPos[subId].x, s_fgBoundaryPos[subId].y);
                }
            }
        }
    }

        

        // random walk
    
}


__global__ void k_fastest_sampleMatch(Byte* d_trimap, uchar4* d_frame, UnknownPixel* unknownPixels, int unknownPixelsCount, Byte* d_dest,
    RandState* d_randState, ushort2* fgBoundaryPos, uint16_t fgBorderCount,
    int width, int height) {
    const int bx = blockDim.x * blockIdx.x + threadIdx.x;
    const int by = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = bx + by * width;
    const int subId = threadIdx.x * threadIdx.y;

    if (id < unknownPixelsCount)
        return;
    RandState localRandState = d_randState[id];
    UnknownPixel pixelInfo = unknownPixels[id];
    const int x = pixelInfo.x;
    const int y = pixelInfo.y;


    __shared__ uchar4 s_sampleColors[256];
    __shared__ int s_cost[256];
    __shared__ int s_bestCost[256];
    __shared__ ushort2 s_fgBoundaryPos[256];

    // load pixels colors
    float3 reducedPixelColor = make_float3(pixelInfo.frameR, pixelInfo.frameG, pixelInfo.frameB);
    float3 bgColor = make_float3(pixelInfo.bgR, pixelInfo.bgG, pixelInfo.bgB);

    // init with random sample
    int sampleIndex = Gpu::Utils::devRand(localRandState) * (fgBorderCount - 1);
    s_fgBoundaryPos[subId] = fgBoundaryPos[sampleIndex];

    s_sampleColors[subId] = d_frame[s_fgBoundaryPos[subId].y * width + s_fgBoundaryPos[subId].x];
    float3 reducedSampleColor = make_float3(UNPACK_V3(s_sampleColors[subId]));

    float alpha = calculateAlpha(reducedPixelColor, reducedSampleColor, bgColor);
    s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
        + sqrtf((s_fgBoundaryPos[subId].x - x) * (s_fgBoundaryPos[subId].x - x) + (s_fgBoundaryPos[subId].y - y) * (s_fgBoundaryPos[subId].y - y));
    s_bestCost[subId] = s_cost[subId];
    uint2 bestPos = make_uint2(s_fgBoundaryPos[subId].x, s_fgBoundaryPos[subId].y);
    // propagation

    for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
            int sampleIndex = x + dx + (y + dy) * width;

            if (d_trimap[sampleIndex] == GRAY) {
                s_fgBoundaryPos[subId] = fgBoundaryPos[sampleIndex];

                s_sampleColors[subId] = s_sampleColors[subId] = d_frame[s_fgBoundaryPos[subId].y * width + s_fgBoundaryPos[subId].x];
                reducedSampleColor = make_float3(UNPACK_V3(s_sampleColors[subId]));

                alpha = calculateAlpha(reducedPixelColor, reducedSampleColor, bgColor);
                s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
                    + sqrtf((s_fgBoundaryPos[subId].x - x) * (s_fgBoundaryPos[subId].x - x) + (s_fgBoundaryPos[subId].y - y) * (s_fgBoundaryPos[subId].y - y));
                if (s_cost[subId] < s_bestCost[subId]) {
                    s_bestCost[subId] = s_cost[subId];
                    bestPos = make_uint2(s_fgBoundaryPos[subId].x, s_fgBoundaryPos[subId].y);
                }
            }
        }
    }



    // random walk

}

__global__ void k_faster_sampleMatch(MattingSample* bestSamples, MattingSample* mattingSamples, Byte* d_trimap, uchar4* d_frame, UnknownPixel* unknownPixels, int unknownPixelsCount, Byte* d_dest,
    RandState* d_randState, ushort2* fgBoundaryPos, uint16_t fgBorderCount,
    int width, int height) {
    const int id = (blockDim.x * blockIdx.x + threadIdx.x) + (blockDim.y * blockIdx.y + threadIdx.y) * width;
    const int subId = threadIdx.x * threadIdx.y;

    if (id < unknownPixelsCount)
        return;
    RandState localRandState = d_randState[id];
    UnknownPixel pixelInfo = unknownPixels[id];
    /*const int x = pixelInfo.x;
    const int y = pixelInfo.y;*/
    MattingSample sample, currentBestSample;


    __shared__ uchar4 s_sampleColors[256];
    __shared__ int s_cost[256];
    __shared__ int s_bestCost[256];
    __shared__ ushort2 s_fgBoundaryPos[256];
    __shared__ uint8_t s_bestAlpha[256];

    // load pixels colors
    float3 reducedPixelColor = make_float3(pixelInfo.frameR, pixelInfo.frameG, pixelInfo.frameB);
    float3 bgColor = make_float3(pixelInfo.bgR, pixelInfo.bgG, pixelInfo.bgB);

    // init with random sample
    int sampleIndex = Gpu::Utils::devRand(localRandState) * (fgBorderCount - 1);
    s_fgBoundaryPos[subId] = fgBoundaryPos[sampleIndex];

    s_sampleColors[subId] = d_frame[s_fgBoundaryPos[subId].y * width + s_fgBoundaryPos[subId].x];
    float3 reducedSampleColor = make_float3(UNPACK_V3(s_sampleColors[subId]));

    float alpha = calculateAlpha(reducedPixelColor, reducedSampleColor, bgColor);
    s_bestAlpha[subId] = alpha * 255;
    s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
        + sqrtf((s_fgBoundaryPos[subId].x - pixelInfo.x) * (s_fgBoundaryPos[subId].x - pixelInfo.x)
            + (s_fgBoundaryPos[subId].y - pixelInfo.y) * (s_fgBoundaryPos[subId].y - pixelInfo.y));
    s_bestCost[subId] = s_cost[subId];

    // TODO init best samples
    // propagation
#pragma unroll
    for (int dx = -1; dx < 2; dx++) {
#pragma unroll
        for (int dy = -1; dy < 2; dy++) {
            // really means id
            sampleIndex = pixelInfo.x + dx + (pixelInfo.y + dy) * width;

            if (d_trimap[sampleIndex] == GRAY) {
                sample = bestSamples[sampleIndex];

                alpha = calculateAlpha(
                    reducedPixelColor,
                    make_float3(sample.R, sample.G, sample.B),
                    bgColor
                );

                s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
                    + sqrtf((sample.x - pixelInfo.x) * (sample.x - pixelInfo.x) 
                        + (sample.y - pixelInfo.y) * (sample.y - pixelInfo.y));
                if (s_cost[subId] < s_bestCost[subId]) {
                    s_bestCost[subId] = s_cost[subId];
                    s_bestAlpha[subId] = alpha * 255;
                    currentBestSample = sample;
                }
            }
        }
    }


    // random walk
    for (float B = 1.f; B * fgBorderCount > 1; B *= 0.5f) {
        sampleIndex = B * (Gpu::Utils::devRand(localRandState) * 2.f - 1.f) * (fgBorderCount - 1);
        if (sampleIndex < 0 || sampleIndex >= fgBorderCount)
            continue;

        sample = mattingSamples[sampleIndex];

        reducedSampleColor = make_float3(sample.R, sample.G, sample.B);
        
        alpha = calculateAlpha(reducedPixelColor, reducedSampleColor, bgColor);
        s_cost[subId] = faster_colorCost(reducedPixelColor, reducedSampleColor, bgColor, alpha)
            + sqrtf((sample.x - pixelInfo.x) * (sample.x - pixelInfo.x)
                + (sample.y - pixelInfo.y) * (sample.y - pixelInfo.y));

        // save best sample
        if (s_cost[subId] < s_bestCost[subId]) {
            s_bestCost[subId] = s_cost[subId];
            s_bestAlpha[subId] = alpha * 255;
            currentBestSample = sample;
        }
    }

    pixelInfo.currentAlpha = s_bestAlpha[subId];
    pixelInfo.bestCost = s_bestCost[subId];
    unknownPixels[id] = pixelInfo;
    bestSamples[id] = currentBestSample;
    d_randState[id] = localRandState;
}

void Gpu::GlobalSampling::matting(Byte * d_image, Byte * d_trimap, Byte* d_background, Byte * d_output)
{
    /*
    UnknownPixel* unknownPixels,
    int* unknownPixelsCount,
    uchar4* frame,
    uint8_t* background*/

    // MATTING
    uint16_t* d_fgBorderX = nullptr;
    uint16_t* d_fgBorderY = nullptr;
    uint32_t* fgBorderPointsCount = nullptr;
    UnknownPixel* d_unknownPixels = nullptr;
    int* d_unknownPixelsCount = 0;
    MattingSample* d_mattingSamples = nullptr;
    int* d_bestSamplesIndexes = nullptr;
    MattingSample** d_bestSamples = nullptr;

    /*NEW---------*/
    ushort2* d_fgBoundaryPos = nullptr;
    /*------------*/

    uint16_t* d_unknownX = nullptr;
    uint16_t* d_unknownY = nullptr;
    uint32_t* d_unknownCount = nullptr;


    float* d_dists = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&fgBorderPointsCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_fgBorderX), m_size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_fgBorderY), m_size * sizeof(uint16_t));

    cudaMalloc(reinterpret_cast<void**>(&d_unknownPixels), m_size * sizeof(UnknownPixel));
    cudaMalloc(reinterpret_cast<void**>(&d_mattingSamples), m_size * sizeof(MattingSample));
    cudaMalloc(reinterpret_cast<void**>(&d_bestSamples), m_size * sizeof(MattingSample*));
    cudaMalloc(reinterpret_cast<void**>(&d_bestSamplesIndexes), m_size * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownPixelsCount), sizeof(int));

    /*NEW-----------*/
    cudaMalloc(reinterpret_cast<void**>(&d_fgBoundaryPos), m_size * sizeof(ushort2));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaArray* tempTextArr;
    cudaMallocArray(&tempTextArr, &channelDesc, m_width, m_height);
    cudaMemcpyToArray(tempTextArr, 0, 0, d_image, sizeof(uchar4) * m_size, cudaMemcpyDeviceToDevice);
    
    d_text_frame.normalized = false;
    d_text_frame.addressMode[0] = cudaAddressModeWrap;
    d_text_frame.addressMode[1] = cudaAddressModeWrap;
    d_text_frame.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(d_text_frame, tempTextArr, channelDesc);
    /*--------------*/

    cudaMemset(reinterpret_cast<void*>(fgBorderPointsCount), 0, sizeof(uint16_t));

    cudaMalloc(reinterpret_cast<void**>(&d_unknownCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownX), m_size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownY), m_size * sizeof(uint16_t));
    cudaMemset(reinterpret_cast<void*>(d_unknownCount), 0, sizeof(uint16_t));


    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };
    uchar4* d_frame = reinterpret_cast<uchar4*>(d_image);

    k_initializeSampleSet<<<dimGrid, dimBlock>>> (
        d_trimap, d_output, 1280, 720, d_fgBorderX, d_fgBorderY, fgBorderPointsCount,
        d_unknownX, d_unknownY, d_unknownCount, d_fgBoundaryPos, 
        d_unknownPixels, d_unknownPixelsCount, d_frame, d_background, d_mattingSamples
    );
    cudaDeviceSynchronize();

    //matting res trimap border fg = 7251 without rand
    uint32_t count = 0, countBorder, unknownPixelsCount;
    cudaMemcpy(&count, d_unknownCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&countBorder, fgBorderPointsCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&unknownPixelsCount, d_unknownPixelsCount, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Unknown count " << countBorder << std::endl;
    int frameSize[] = { m_width, m_height };
    cudaMemcpyToSymbol(c_frameSize, frameSize, sizeof(int) * 2);

    std::cout << "Uknown" << unknownPixelsCount << std::endl;

    k_initBestSamples << <dimGrid, dimBlock >> > (d_mattingSamples, countBorder, d_trimap, d_bestSamples, 
        d_bestSamplesIndexes, m_d_randStates, m_width, m_height);
    cudaDeviceSynchronize();
    
    k_sampleMatch<<<dimGrid, dimBlock>>>(d_trimap, d_frame, d_background, d_output, m_d_randStates, 
        d_fgBoundaryPos, countBorder, m_width, m_height);
    cudaDeviceSynchronize();
    k_faster_sampleMatch << <dimGrid, dimBlock >> > (d_mattingSamples, d_mattingSamples, d_trimap, d_frame, d_unknownPixels,
        unknownPixelsCount, d_output, m_d_randStates,
        d_fgBoundaryPos, countBorder, m_width, m_height);
    cudaDeviceSynchronize();

    k_fastest_sampleMatch << <dimGrid, dimBlock >> > (d_trimap, d_frame, d_unknownPixels,
         unknownPixelsCount, d_output, m_d_randStates,
        d_fgBoundaryPos, countBorder, m_width, m_height);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    std::cout << cudaGetErrorName(err);
}