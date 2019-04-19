#include "guidedfilter.cuh"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "helper_math.h"
#include "../../common/config.h"


GuidedFilter::GuidedFilter() {
    m_size = FRAME_WIDTH * FRAME_HEIGHT;

    cudaMalloc(reinterpret_cast<void**>(&m_d_meanR), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanG), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanB), m_size * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_varRR), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varRG), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varRB), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varGG), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varGB), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_varBB), m_size * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_meanP), m_size * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_meanPR), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanPG), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_meanPB), m_size * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&m_d_AR), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_AG), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_AB), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&m_d_B), m_size * sizeof(float));
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

    cudaFree(m_d_AR);
    cudaFree(m_d_AG);
    cudaFree(m_d_AB);
    cudaFree(m_d_B);
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
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    prepareData << <dimGrid, dimBlock >> > (d_frame, d_alphaMask, m_d_meanR, m_d_meanG, m_d_meanB, 
        m_d_varRR, m_d_varRG, m_d_varRB, m_d_varGG, m_d_varGB, m_d_varBB,
        m_d_meanPR, m_d_meanPG, m_d_meanPB
    );
    cudaDeviceSynchronize();

    float* temp;
    float* temp16;
    float* tempF;
    cudaMalloc(reinterpret_cast<void**>(&temp), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&temp16), m_size * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&tempF), m_size * sizeof(float));

    // mean R
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_meanR, temp, 2);
    // mean G
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_meanG, temp, 2);
    // mean B
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_meanB, temp, 2);

    // var RR
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_varRR, temp16, 2);
    // var RG
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_varRG, temp16, 2);
    // var RB
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_varRB, temp16, 2);
    // var GG
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_varGG, temp16, 2);
    // var GB
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_varGB, temp16, 2);
    // var BB
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_varBB, temp16, 2);

    // mean P
    //Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, d_alphaMask, , temp, 2);
    Gpu::Utils::k_boxFilter_sep_x<uint8_t, float> << <dimGrid, dimBlock >> > (d_alphaMask, temp, 2);
    cudaDeviceSynchronize();
    Gpu::Utils::k_boxFilter_sep_y<float, float> << <dimGrid, dimBlock >> > (temp, m_d_meanP, 2);
    cudaDeviceSynchronize();

    // var PR
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_meanPR, temp16, 2);
    // var PG
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_meanPG, temp16, 2);
    // var PB
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_meanPB, temp16, 2);

    /*uint8_t* meanR, uint8_t* meanG, uint8_t* meanB, uint8_t* meanP,
    uint16_t* varRR, uint16_t* varRG, uint16_t* varRB,
    uint16_t* varGG, uint16_t* varGB, uint16_t* varBB,
    uint16_t* meanPR, uint16_t* meanPG, uint16_t* meanPB,
    float* AR, float* AG, float* AB, float* B, float eps*/
    semi_compute << <dimGrid, dimBlock >> > (m_d_meanR, m_d_meanG, m_d_meanB, m_d_meanP,
        m_d_varRR, m_d_varRG, m_d_varRB, m_d_varGG, m_d_varGB, m_d_varBB,
        m_d_meanPR, m_d_meanPG, m_d_meanPB, m_d_AR, m_d_AG, m_d_AB, m_d_B, 1e-5f
    );
    cudaDeviceSynchronize();

    // A
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_AR, tempF, 2);
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_AG, tempF, 2);
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_AB, tempF, 2);
    Gpu::Utils::boxFilter<float>(dimGrid, dimBlock, m_d_B, tempF, 2);

    /*uchar4* frame, float* AR, float* AG, float* AB, float* B, uint8_t* dest*/
    finish_compute << <dimGrid, dimBlock >> > (d_frame, m_d_AR, m_d_AG, m_d_AB, m_d_B, d_output);
}

