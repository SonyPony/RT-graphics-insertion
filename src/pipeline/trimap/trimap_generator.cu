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
    
    if (thresholded)
        dest[id] = thresholded * 128;
    destCopy[id] = thresholded * 255;
}

__global__ void k_assembleTrimap(uint8_t* currentTrimap, uint8_t* dilatatedEdges) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    if (currentTrimap[id] == 0 && dilatatedEdges[id] == 255)
        currentTrimap[id] = 128;

}

void TrimapGenerator::generate(uint8_t* d_segmentationMask, uint8_t * d_dest)
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    Gpu::Utils::gradients(dimGrid, dimBlock, d_segmentationMask, m_d_temp16, m_d_grads);

    k_thresholdGrads << <dimGrid, dimBlock >> > (m_d_grads, d_dest, m_d_dilatatedEdges, 10);
    FilterDilation(m_d_dilatatedEdges, m_d_temp, FRAME_WIDTH, FRAME_HEIGHT, 2);
    k_assembleTrimap << <dimGrid, dimBlock >> > (d_dest, m_d_dilatatedEdges);
}
