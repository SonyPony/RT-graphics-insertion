#include "globalmatting.cuh"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "helper_math.h"

#include "../../common/config.h"

#define GRAY 128
using namespace std;

__constant__ int c_frameSize[2];

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

__global__ void k_initializeUnknown(
    uint8_t* trimap, 
    UnknownPixel* unknownPixels,
    int* unknownPixelsCount, 
    uchar4* frame,
    uint8_t* background
    ) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int yOffset = y * FRAME_WIDTH;
    int i;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1 || trimap[x + y * FRAME_WIDTH] != GRAY)
        return;

    i = atomicAdd(unknownPixelsCount, 1);
    UnknownPixel up;
    up.bgR = background[(yOffset + x)];
    up.bgG = background[(yOffset + x) + (FRAME_HEIGHT * FRAME_WIDTH)];
    up.bgB = background[(yOffset + x) + 2 * (FRAME_HEIGHT * FRAME_WIDTH)];

    uchar4 framePixel = frame[yOffset + x];
    up.frameR = framePixel.x;
    up.frameG = framePixel.y;
    up.frameB = framePixel.z;
    up.bestCost = FLT_MAX;

    up.x = x;
    up.y = y;

    unknownPixels[i] = up;
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
    uchar4 framePixel = frame[FRAME_WIDTH * y + x];

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

inline __device__ float calculateAlpha(float3 I,
    float3 F, float3 B) {

    float3 numerator = (I - B) * (F - B);
    float3 denominator = (F - B) * (F - B);

    return clamp(Gpu::Utils::sum(numerator) / (1e-6f + Gpu::Utils::sum(denominator)), 0.f, 1.f);
}

inline __device__ float faster_colorCost(float3 I,
    float3 F, float3 B, float alpha) {

    F = I - (F * alpha + B * (1.f - alpha));
    return norm3df(F.x, F.y, F.z);
}

inline __device__ float distanceColor(uint2 pixelPos, uint2 samplePos) {
    const float2 fPixelPos = make_float2(UNPACK_V2(pixelPos));
    const float2 fsamplePos = make_float2(UNPACK_V2(samplePos));

    return length(fsamplePos - fPixelPos);
}

/*__global__ void k_convolve_x(uint8_t* input, uint8_t* output) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x == 0 || x == FRAME_WIDTH - 1 || y == 0 || y == FRAME_HEIGHT - 1)
        return;

    int kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    float outputValue = 0.f;

    for (int i = -1; i <= 1; i++) { //x
        for (int j = -1; j <= 1; j++) { //y
            outputValue += kernel[j][i] * input[x + i + (y + j) * FRAME_WIDTH];
        }
    }

    output[x + y * FRAME_WIDTH] = (int)outputValue;
}*/

// TODO del frame
__global__ void k_faster_sampleMatch(
    int* bestSamplesIndexes , 
    MattingSample* mattingSamples, 
    Byte* d_trimap, uchar4* d_frame, 
    UnknownPixel* unknownPixels, int unknownPixelsCount, Byte* d_dest,
    RandState* d_randState, uint16_t fgBorderCount) {
    const int id = (blockDim.x * blockIdx.x + threadIdx.x) + (blockDim.y * blockIdx.y + threadIdx.y) * FRAME_WIDTH;
    const int subId = threadIdx.x * threadIdx.y;

    if (id >= unknownPixelsCount)
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
            //break;
            int n_id = dx + dy * 1280;

            if (d_trimap[n_id] != GRAY)
                continue;

            sampleIndex = bestSamplesIndexes[n_id];
            sample = mattingSamples[sampleIndex];
            F = make_float3(sample.R, sample.G, sample.B);

            alpha = calculateAlpha(I, F, B);

            s_cost[subId] = 10 * faster_colorCost(I, F, B, alpha)
                + norm3df((float)sample.x - (float)pixelInfo.x, (float)sample.y - (float)pixelInfo.y, 0.f);

            if (s_cost[subId] < pixelInfo.bestCost) {
                pixelInfo.bestCost = s_cost[subId];
                pixelInfo.currentAlpha = alpha * 255;
                bestSampleIndex = sampleIndex;
            }
            
        }
    }


    // random walk
    for (float BK = 1.f; BK * fgBorderCount > 1.f; BK *= 0.5f) {
        sampleIndex = bestSampleIndex + BK * (Gpu::Utils::devRand(localRandState) * 2.f - 1.f) * (fgBorderCount - 1);
        if (sampleIndex < 0 || sampleIndex >= fgBorderCount)
            continue;
        //break;
        sample = mattingSamples[sampleIndex];

        F = make_float3(sample.R, sample.G, sample.B);

        alpha = calculateAlpha(I, F, B);
        s_cost[subId] = 10*faster_colorCost(I, F, B, alpha)
            + norm3df((float)sample.x - (float)pixelInfo.x, (float)sample.y - (float)pixelInfo.y, 0.f);

        // save best sample
        if (s_cost[subId] < pixelInfo.bestCost) {
            pixelInfo.bestCost = s_cost[subId];
            pixelInfo.currentAlpha = alpha * 255;
            bestSampleIndex = sampleIndex;
        }
    }

    //pixelInfo.currentAlpha = 255;
    unknownPixels[id] = pixelInfo;
    d_randState[id] = localRandState;
    bestSamplesIndexes[pixelInfo.x + pixelInfo.y * 1280] = bestSampleIndex;
}

#include<chrono>

__global__ void k_renderMatting(uint8_t* alphaMask, UnknownPixel* unknownPixels, int unknownPixelsCount, uint8_t* trimap) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = y * FRAME_WIDTH + x;

    if (trimap[id] == 255)
        alphaMask[id] = 255;

    if (id >= unknownPixelsCount)
        return;

    const auto pixelInfo = unknownPixels[id];
    const int pixelId = pixelInfo.x + pixelInfo.y * FRAME_WIDTH;
    alphaMask[pixelId] = pixelInfo.currentAlpha;
}

void Gpu::GlobalSampling::matting(Byte * d_image, Byte * d_trimap, Byte* d_background, Byte * d_output)
{
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


    /*--------------*/

    cudaMemset(reinterpret_cast<void*>(fgBorderPointsCount), 0, sizeof(uint16_t));

    cudaMalloc(reinterpret_cast<void**>(&d_unknownCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownX), m_size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownY), m_size * sizeof(uint16_t));
    cudaMemset(reinterpret_cast<void*>(d_unknownCount), 0, sizeof(uint16_t));


    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };
    uchar4* d_frame = reinterpret_cast<uchar4*>(d_image);
    uint32_t count = 0, countBorder, unknownPixelsCount;

    k_initializeSampleSet<<<dimGrid, dimBlock>>> (
        d_trimap, d_output, 1280, 720, d_fgBorderX, d_fgBorderY, fgBorderPointsCount,
        d_unknownX, d_unknownY, d_unknownCount, d_fgBoundaryPos, 
        d_unknownPixels, d_unknownPixelsCount, d_frame, d_background, d_mattingSamples
    );

    k_initializeUnknown << <dimGrid, dimBlock >> > (
        d_trimap,
        d_unknownPixels,
        d_unknownPixelsCount,
        d_frame,
        d_background
        );

    cudaDeviceSynchronize();
    cudaMemcpy(&unknownPixelsCount, d_unknownPixelsCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Uknown" << unknownPixelsCount << std::endl;
    

    //matting res trimap border fg = 7251 without rand
    
    cudaMemcpy(&count, d_unknownCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&countBorder, fgBorderPointsCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    

    std::cout << "Unknown count " << countBorder << std::endl;
    int frameSize[] = { m_width, m_height };
    cudaMemcpyToSymbol(c_frameSize, frameSize, sizeof(int) * 2);

    

    k_initBestSamples << <dimGrid, dimBlock >> > (d_mattingSamples, countBorder, d_trimap, d_bestSamples, 
        d_bestSamplesIndexes, m_d_randStates, m_width, m_height);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; i++) {
        k_faster_sampleMatch << <dimGrid, dimBlock >> > (d_bestSamplesIndexes, d_mattingSamples, d_trimap, d_frame, d_unknownPixels,
            unknownPixelsCount, d_output, m_d_randStates,
            countBorder);

        /*k_fast_sampleMatch << <dimGrid, dimBlock >> > (d_bestSamples, d_mattingSamples, d_trimap, d_frame, d_unknownPixels,
            unknownPixelsCount, d_output, m_d_randStates,
            countBorder, m_width, m_height);*/
        //cudaDeviceSynchronize();
    }

    cudaMemset(d_output, 0, m_size);
    k_renderMatting << <dimGrid, dimBlock >> > (d_output, d_unknownPixels, unknownPixelsCount, d_trimap);

    // test convolve
    uint8_t* temp = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&temp), m_size);
    cudaMemcpy(temp, d_output, m_size, cudaMemcpyDeviceToDevice);

    //Gpu::Utils::boxFilter<uint8_t>(dimGrid, dimBlock, d_output, temp, 2, false);

    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    m_guidedFilter.filter(d_frame, temp, d_output);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(t2 - t1).count();

    cout << "time" << duration << std::endl;;
    //-----------

    cudaDeviceSynchronize();
    
    auto err = cudaGetLastError();
    std::cout << cudaGetErrorName(err);
}