#include "vibe.cuh"
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "../../common/config.h"


__global__ void k_segment(uint8_t* input, uint8_t* model, uint8_t* dest, int size, int currentSample, RandState* randState) {
    constexpr int stride = 4;    // 4 bytes strides
    const int id = (blockDim.x * blockIdx.x + threadIdx.x);
    const int pixel_i = id * stride;

    const float localInput[3] = { (float)input[pixel_i] , (float)input[pixel_i + 1] , (float)input[pixel_i + 2] };

    int count = 0;
    for (int j = 0; j < 20; j++) {
        const int sample_i = size * j * 3;
        float distance = norm3df(
            localInput[0] - model[id + sample_i],
            localInput[1] - model[id + sample_i + size],
            localInput[2] - model[id + sample_i + size * 2]
        );

        if (distance < 20)
            count++;
    }

    const int isForeground = (count < 2);
    dest[id] = isForeground * 255;

    // update
    RandState localRandState = randState[id];
    uint8_t rand = Gpu::Utils::devRand(localRandState) * 16;
    if (rand == 0) {
        for (int i = 0; i < 3; i++)
            model[id + currentSample + i * size] = localInput[i];
    }

    rand = Gpu::Utils::devRand(localRandState) * 16;
    // todo change to something like stencil?
    if (rand == 0) {
        for (int i = 0; i < 3; i++)
            model[id + currentSample + i * size] = localInput[i];
    }

    randState[id] = localRandState;
}

__global__ void k_initBackgroundModelSamples(uint8_t* input, uint8_t* dest, uint8_t samplesCount, int size) {
    constexpr int stride = 4;    // 4 bytes strides
    const int rgba_i = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);
    const int pixelffset = (i % size) * 3;

    for (int j = 0; j < samplesCount; j++) {
        for (int channel = 0; channel < 3; channel++)
            dest[j * size * 3 + i + channel * size] = input[rgba_i + channel];
    }
}

Gpu::ViBe::ViBe(int width, int height)
    : m_width{ width }, m_height(height), m_size(height * width) {
    const int modelSampleSize = m_size * Config::CHANNELS_COUNT_BG_SAMPLE;
    const int modelSize = modelSampleSize * ViBe::SAMPLE_COUNT;

    // init rand states for vibe update
    Gpu::Utils::generateRandStates(&m_d_randState, m_size);

    // alloc background model on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_bgModel), modelSize);
}

Gpu::ViBe::~ViBe()
{
    cudaFree(m_d_bgModel);
    cudaFree(m_d_randState);
}

void Gpu::ViBe::initialize(Byte* backgroundModel) {
    const int sampleSize = m_size * Config::CHANNELS_COUNT_INPUT;
    uint8_t* d_bgInit = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_bgInit), sampleSize);
    cudaMemcpy(d_bgInit, backgroundModel, sampleSize, cudaMemcpyHostToDevice);

    k_initBackgroundModelSamples<<<900, 1024 >>> (d_bgInit, m_d_bgModel, ViBe::SAMPLE_COUNT, m_width * m_height);
}

void Gpu::ViBe::segment(Byte* d_input, Byte* d_dest) {
    k_segment<<<1800, 512>>> (d_input, m_d_bgModel, d_dest, m_size, 0, m_d_randState);

    auto err = cudaGetLastError();
    std::cout << cudaGetErrorName(err);
}