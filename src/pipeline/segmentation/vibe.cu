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
    uint8_t rand = Gpu::Utils::devRand(localRandState) * ViBe::PHI;
    if (rand == 0 && !isForeground) {
        model[id + currentSample * FRAME_SIZE] = inputPixel;
    }

    randState[id] = localRandState;
}

__global__ void k_updateNeighbours(uint8_t* segMask, uchar4* model, int currentSample, RandState* randState) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uchar4 inputPixel = model[id + currentSample * FRAME_SIZE];

    RandState localRandState = randState[id];
    constexpr int dx[4] = {-1, 0, 1, 0 };
    constexpr int dy[4] = { 0, 1, 0, -1 };

    const uint8_t neighbourIndex = Gpu::Utils::devRand(localRandState) * 4;
    const uint8_t updateIProb = Gpu::Utils::devRand(localRandState) * ViBe::PHI;

    if (updateIProb == 0 && segMask[id] == BACKGROUND) {
        const int ndx = dx[neighbourIndex];
        const int ndy = dy[neighbourIndex];

        if (x + ndx < 0 || x + ndx > FRAME_WIDTH - 1 || y + ndy < 0 || y + ndy > FRAME_HEIGHT - 1)
            return;
        model[x + ndx + (y + ndy) * FRAME_WIDTH + currentSample * FRAME_SIZE] = inputPixel;
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
    m_sampleIndex = 0;
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

void ViBe::initialize(uchar4* d_backgroundModel) {
    k_initBackgroundModelSamples<<<DIM_GRID, DIM_BLOCK >>> (d_backgroundModel, m_d_bgModel);
}

uchar4* ViBe::segment(uchar4* d_input, uint8_t* d_dest) {
    k_segment<<<DIM_GRID, DIM_BLOCK >>> (d_input, m_d_bgModel, d_dest, m_sampleIndex, m_d_randState);
    k_updateNeighbours << <DIM_GRID, DIM_BLOCK >> > (d_dest, m_d_bgModel, m_sampleIndex, m_d_randState);

    m_sampleIndex = (m_sampleIndex + 1) % ViBe::SAMPLE_COUNT;

    return m_d_bgModel;
}