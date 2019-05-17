#include "guidedfilter.cuh"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "helper_math.h"
#include "../../common/config.h"


GuidedFilter::GuidedFilter() {
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanR), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanG), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanB), FRAME_SIZE * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_varRR), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varRG), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varRB), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varGG), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varGB), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varBB), FRAME_SIZE * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_meanP), FRAME_SIZE * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_meanPR), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanPG), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanPB), FRAME_SIZE * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_AR), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_AG), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_AB), FRAME_SIZE * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_B), FRAME_SIZE * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_tempF), FRAME_SIZE * sizeof(float));
}

GuidedFilter::~GuidedFilter() {
    cudaFree(m_d_meanR);
    cudaFree(m_d_meanG);
    cudaFree(m_d_meanB);

    cudaFree(m_d_varRR);
    cudaFree(m_d_varRG);
    cudaFree(m_d_varRB);
    cudaFree(m_d_varGG);
    cudaFree(m_d_varGB);
    cudaFree(m_d_varBB);

    cudaFree(m_d_meanP);
    cudaFree(m_d_meanPR);
    cudaFree(m_d_meanPG);
    cudaFree(m_d_meanPB);

    cudaFree(m_d_AR);
    cudaFree(m_d_AG);
    cudaFree(m_d_AB);
    cudaFree(m_d_B);

    cudaFree(m_d_tempF);
}

__global__ void prepareData(
    uchar4* frame, uint8_t* alphaMask,
    float* meanR, float* meanG, float* meanB,
    float* varRR, float* varRG, float* varRB,
    float* varGG, float* varGB, float* varBB,
    float* varPR, float* varPG, float* varPB
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = y * FRAME_WIDTH + x;

    // TODO skip borders??

    const uchar4 pixelColor = frame[id];
    const int alpha = alphaMask[id];

    meanR[id] = pixelColor.x;
    meanG[id] = pixelColor.y;
    meanB[id] = pixelColor.z;

    varRR[id] = static_cast<int>(pixelColor.x) * pixelColor.x;
    varRG[id] = static_cast<int>(pixelColor.x) * pixelColor.y;
    varRB[id] = static_cast<int>(pixelColor.x) * pixelColor.z;
    varGG[id] = static_cast<int>(pixelColor.y) * pixelColor.y;
    varGB[id] = static_cast<int>(pixelColor.y) * pixelColor.z;
    varBB[id] = static_cast<int>(pixelColor.z) * pixelColor.z;

    varPR[id] = alpha * pixelColor.x;
    varPG[id] = alpha * pixelColor.y;
    varPB[id] = alpha * pixelColor.z;
}

__global__ void semi_compute(float* meanR, float* meanG, float* meanB, float* meanP,
    float* varRR, float* varRG, float* varRB,
    float* varGG, float* varGB, float* varBB,
    float* meanPR, float* meanPG, float* meanPB,
    float* AR, float* AG, float* AB, float* B, float eps
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = y * FRAME_WIDTH + x;

    const float l_meanR = meanR[id];
    const float l_meanG = meanG[id];
    const float l_meanB = meanB[id];
    const float l_meanP = meanP[id];

    const float l_varRR = static_cast<float>(varRR[id]) - l_meanR * l_meanR + eps;
    const float l_varRG = static_cast<float>(varRG[id]) - l_meanR * l_meanG;
    const float l_varRB = static_cast<float>(varRB[id]) - l_meanR * l_meanB;
    const float l_varGG = static_cast<float>(varGG[id]) - l_meanG * l_meanG + eps;
    const float l_varGB = static_cast<float>(varGB[id]) - l_meanG * l_meanB;
    const float l_varBB = static_cast<float>(varBB[id]) - l_meanB * l_meanB + eps;

    float invRR = l_varGG * l_varBB - l_varGB * l_varGB;
    float invRG = l_varGB * l_varRB - l_varRG * l_varBB;
    float invRB = l_varRG * l_varGB - l_varGG * l_varRB;
    float invGG = l_varRR * l_varBB - l_varRB * l_varRB;
    float invGB = l_varRB * l_varRG - l_varRR * l_varGB;
    float invBB = l_varRR * l_varGG - l_varRG * l_varRG;

    const float convDet = invRR * l_varRR + invRG * l_varRG + invRB * l_varRB;

    invRR /= convDet;
    invRG /= convDet;
    invRB /= convDet;
    invGG /= convDet;
    invGB /= convDet;
    invBB /= convDet;

    float covPR = meanPR[id] - l_meanR * l_meanP;
    float covPG = meanPG[id] - l_meanG * l_meanP;
    float covPB = meanPB[id] - l_meanB * l_meanP;

    // TODO store
    float l_AR = invRR * covPR + invRG * covPG + invRB * covPB;
    float l_AG = invRG * covPR + invGG * covPG + invGB * covPB;
    float l_AB = invRB * covPR + invGB * covPG + invBB * covPB;

    float l_B = l_meanP - l_AR * l_meanR - l_AG * l_meanG - l_AB * l_meanB;

    AR[id] = l_AR;
    AG[id] = l_AG;
    AB[id] = l_AB;
    B[id] = l_B;
}

__global__ void finish_compute(uchar4* frame, float* AR, float* AG, float* AB, float* B, uint8_t* dest) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = y * FRAME_WIDTH + x;

    const uchar4 pixelColor = frame[id];
    dest[id] = clamp(AR[id] * pixelColor.x + AG[id] * pixelColor.y + AB[id] * pixelColor.z + B[id], 0.f, 255.f);
}

void GuidedFilter::filter(uchar4 * d_frame, uint8_t * d_alphaMask, uint8_t * d_output)
{
    prepareData << <DIM_GRID, DIM_BLOCK >> > (d_frame, d_alphaMask, m_d_meanR, m_d_meanG, m_d_meanB,
        m_d_varRR, m_d_varRG, m_d_varRB, m_d_varGG, m_d_varGB, m_d_varBB,
        m_d_meanPR, m_d_meanPG, m_d_meanPB
    );

    // mean R
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_meanR, m_d_tempF, 2);
    // mean G
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_meanG, m_d_tempF, 2);
    // mean B
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_meanB, m_d_tempF, 2);

    // var RR
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_varRR, m_d_tempF, 2);
    // var RG
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_varRG, m_d_tempF, 2);
    // var RB
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_varRB, m_d_tempF, 2);
    // var GG
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_varGG, m_d_tempF, 2);
    // var GB
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_varGB, m_d_tempF, 2);
    // var BB
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_varBB, m_d_tempF, 2);

    // mean P
    Gpu::Utils::k_boxFilter_sep_x<uint8_t, float> << <DIM_GRID, DIM_BLOCK >> > (d_alphaMask, m_d_tempF, 2);
    Gpu::Utils::k_boxFilter_sep_y<< <DIM_GRID, DIM_BLOCK >> > (m_d_tempF, m_d_meanP, 2);

    // var PR
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_meanPR, m_d_tempF, 2);
    // var PG
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_meanPG, m_d_tempF, 2);
    // var PB
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_meanPB, m_d_tempF, 2);

    semi_compute << <DIM_GRID, DIM_BLOCK >> > (m_d_meanR, m_d_meanG, m_d_meanB, m_d_meanP,
        m_d_varRR, m_d_varRG, m_d_varRB, m_d_varGG, m_d_varGB, m_d_varBB,
        m_d_meanPR, m_d_meanPG, m_d_meanPB, m_d_AR, m_d_AG, m_d_AB, m_d_B, 1e-5f
    );

    // A
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_AR, m_d_tempF, 2);
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_AG, m_d_tempF, 2);
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_AB, m_d_tempF, 2);
    Gpu::Utils::boxFilter<float>(DIM_GRID, DIM_BLOCK, m_d_B, m_d_tempF, 2);

    finish_compute << <DIM_GRID, DIM_BLOCK >> > (d_frame, m_d_AR, m_d_AG, m_d_AB, m_d_B, d_output);
}

