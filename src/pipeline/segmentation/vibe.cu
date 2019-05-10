#include "vibe.cuh"
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "../../common/config.h"


__global__ void k_segment(uchar4* input, uchar4* model, uint8_t* dest, int currentSample, RandState* randState) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uchar4 inputPixel = input[id];

    int count = 0;
    for (int j = 0; j < ViBe::SAMPLE_COUNT; j++) {
        const uchar4 bgSamplePixel = model[id + j * FRAME_SIZE];

        const float distance = norm3df(
            static_cast<float>(inputPixel.x) - bgSamplePixel.x,
            static_cast<float>(inputPixel.y) - bgSamplePixel.y,
            static_cast<float>(inputPixel.z) - bgSamplePixel.z
        );

        if (distance < ViBe::COLOR_RADIUS)
            count++;
    }

    const int isForeground = (count < 2);
    dest[id] = isForeground * FOREGROUND;

    // update
    RandState localRandState = randState[id];
    uint8_t rand = Gpu::Utils::devRand(localRandState) * 16;
    if (rand == 0 && !isForeground) {
        for (int i = 0; i < 3; i++)
            model[id + currentSample + i * FRAME_SIZE] = inputPixel;
    }

    rand = Gpu::Utils::devRand(localRandState) * 16;
    // todo change to something like stencil?
    if (rand == 0 && !isForeground) {
        for (int i = 0; i < 3; i++)
            model[id + currentSample + i * FRAME_SIZE] = inputPixel;
    }

    randState[id] = localRandState;
}

__global__ void k_initBackgroundModelSamples(uchar4* input, uchar4* dest) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uchar4 inputPixel = input[id];

    for (int sampleIndex = 0; sampleIndex < ViBe::SAMPLE_COUNT; sampleIndex++)
        dest[id + sampleIndex * FRAME_SIZE] = inputPixel;
}

ViBe::ViBe(uint8_t* d_tempBuffer) {
    m_d_temp = d_tempBuffer;
    // init rand states for vibe update
    Gpu::Utils::generateRandStates(&m_d_randState, FRAME_SIZE);

    // alloc background model on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_bgModel), ViBe::SAMPLE_COUNT * FRAME_SIZE * sizeof(uchar4));
}

ViBe::~ViBe()
{
    cudaFree(m_d_bgModel);
    cudaFree(m_d_randState);
}

void ViBe::initialize(uint8_t* backgroundModel) {
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };
    

    const int sampleSize = FRAME_SIZE * Config::CHANNELS_COUNT_INPUT;

    cudaMemcpy(m_d_temp, backgroundModel, FRAME_SIZE * sizeof(uchar4), cudaMemcpyHostToDevice);
    uchar4* d_bgInit = reinterpret_cast<uchar4*>(m_d_temp);

    k_initBackgroundModelSamples<<<dimGrid, dimBlock>>> (d_bgInit, m_d_bgModel);
}

uchar4* ViBe::segment(uchar4* d_input, uint8_t* d_dest) {
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    k_segment<<<dimGrid, dimBlock>>> (d_input, m_d_bgModel, d_dest, 0, m_d_randState);

    auto err = cudaGetLastError();
    std::cout << cudaGetErrorName(err);

    return m_d_bgModel;
}