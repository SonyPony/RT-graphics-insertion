#include "trimap_generator.cuh"
#include <stdint.h>
#include <math.h>
#include "helper_math.h"
#include "../../util/gpuutil.cuh"
#include "../morphology/erosionFuncTemplate.h"

TrimapGenerator::TrimapGenerator()
{
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp16), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));
    cudaMalloc(reinterpret_cast<void**>(&m_d_grads), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp), FRAME_WIDTH * FRAME_HEIGHT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_dilatatedEdges), FRAME_WIDTH * FRAME_HEIGHT);
}

TrimapGenerator::~TrimapGenerator()
{
    cudaFree(m_d_temp16);
    cudaFree(m_d_grads);
    cudaFree(m_d_temp);
    cudaFree(m_d_dilatatedEdges);
}

__global__ void k_thresholdGrads(short2* grads, uint8_t* dest, uint8_t* destCopy, int T) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const short2 gradient = grads[id];
    const uint8_t thresholded = norm3df(gradient.x, gradient.y, 0.f) > T;
    
    dest[id] = thresholded * 128;
    destCopy[id] = thresholded * 255;
}

__global__ void k_assembleTrimap(uint8_t* currentTrimap, uint8_t* dilatatedEdges, uint8_t* segmentationMask) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uint8_t trimapVal = currentTrimap[id];
    const uint8_t edgesVal = dilatatedEdges[id];
    const uint8_t segVal = segmentationMask[id];

    if (trimapVal == 0 && edgesVal == 255 && segVal == 0)
        currentTrimap[id] = 128;
    else if (segVal == 255 && trimapVal == 0)
        currentTrimap[id] = 255;
}

void TrimapGenerator::generate(uint8_t* d_segmentationMask, uint8_t * d_dest)
{
    Gpu::Utils::gradients(DIM_GRID, DIM_BLOCK, d_segmentationMask, m_d_temp16, m_d_grads);

    k_thresholdGrads << <DIM_GRID, DIM_BLOCK >> > (m_d_grads, d_dest, m_d_dilatatedEdges, 10);
    FilterDilation(m_d_dilatatedEdges, m_d_temp, FRAME_WIDTH, FRAME_HEIGHT, 3);
    k_assembleTrimap << <DIM_GRID, DIM_BLOCK >> > (d_dest, m_d_dilatatedEdges, d_segmentationMask);
}
