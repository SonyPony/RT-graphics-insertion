#include "globalmatting.cuh"
#include <stdint.h>
#include <math.h>
#include <limits>

using namespace std;

// MATTING
    /*uint16_t* d_fgBorderX = nullptr;
    uint16_t* d_fgBorderY = nullptr;
    uint32_t* fgBorderPointsCount = nullptr;

    uint16_t* d_unknownX = nullptr;
    uint16_t* d_unknownY = nullptr;
    uint32_t* d_unknownCount = nullptr;
    float* d_dists = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&fgBorderPointsCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_fgBorderX), size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_fgBorderY), size * sizeof(uint16_t));
    cudaMemset(reinterpret_cast<void*>(fgBorderPointsCount), 0, sizeof(uint16_t));

    cudaMalloc(reinterpret_cast<void**>(&d_unknownCount), sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownX), size * sizeof(uint16_t));
    cudaMalloc(reinterpret_cast<void**>(&d_unknownY), size * sizeof(uint16_t));
    cudaMemset(reinterpret_cast<void*>(d_unknownCount), 0, sizeof(uint16_t));


    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    dim3 dimGrid2{ 40, 90 };
    dim3 dimBlock2{ 32, 8 };

    matting <<<dimGrid, dimBlock>>> (
        d_trimap, d_dest, 1280, 720, d_fgBorderX, d_fgBorderY, fgBorderPointsCount,
        d_unknownX, d_unknownY, d_unknownCount
    );

    //matting res trimap border fg = 7251 without rand
    uint32_t count = 0, countBorder;
    cudaMemcpy(&count, d_unknownCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&countBorder, fgBorderPointsCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaMalloc(reinterpret_cast<void**>(&d_dists), size * sizeof(float));
    dim3 gridDim{
        (unsigned int)ceil(count / 256.f)
    };*/

    /*sampleMatch <<<gridDim, 256>>> (d_trimap, d_unknownX, d_unknownY, count,
        d_fgBorderX, d_fgBorderY, countBorder, d_dists);*/

        //const int ss = 100;

        /*sampleMatchWhole<<<dimGrid, dimBlock >>> (d_trimap,
            d_fgBorderX, d_fgBorderY, countBorder, d_dists);*/

__global__ void matting(
    uint8_t* trimap,
    uint8_t* dest,
    int width,
    int height,
    uint16_t* fgBorderX,
    uint16_t* fgBorderY,
    uint32_t* fgBorderCounter,

    uint16_t* unknownX,
    uint16_t* unknownY,
    uint32_t* unkownCounter
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int yOffset = y * width;

    constexpr uint8_t GRAY = 128;
    if (trimap[x + yOffset] == GRAY)
        atomicAdd(fgBorderCounter, 1);

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || trimap[x + y * width] != 255)
        return;

    //dest[x + y * width] = 0;
    if (trimap[x + yOffset + 1] == GRAY || trimap[x + yOffset - 1] == GRAY
        || trimap[x + yOffset + width] == GRAY || trimap[x + yOffset - width] == GRAY) {
        int i = atomicAdd(fgBorderCounter, 1);
        fgBorderX[i] = x;
        fgBorderY[i] = y;
        dest[x + y * width] = 180;

    } // TODO rand vzorky?

    if (trimap[x + yOffset] == GRAY) {
        int i = atomicAdd(unkownCounter, 1);
        unknownX[i] = x;
        unknownY[i] = y;
    }

}

__global__ void sampleMatch(uint8_t* trimap, uint16_t* unknownX, uint16_t* unknownY,
    uint32_t unknownCount, uint16_t* fgBorderX, uint16_t* fgBorderY, uint32_t fgBorderCount,
    float* dists) {
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= unknownCount)
        return;

    float minDist = FLT_MAX;
    for (int i = 0; i < fgBorderCount; i++) {
        float distance = norm3df(
            (int)unknownX[id] - fgBorderX[i],
            (int)unknownY[id] - fgBorderY[i], 0.f
        );
        minDist = min(minDist, distance);
    }

    dists[id] = minDist;
}

//using namespace cooperative_groups;

__global__ void sampleMatchWhole(uint8_t* trimap, uint16_t* fgBorderX,
    uint16_t* fgBorderY, const uint32_t fgBorderCount,
    float* dists) {

    //grid_group g = this_grid();
    //g.sync();
    //grid_group g = this_grid();

    //extern __shared__ int borderPos[];

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * 1280; // TODO size

    uint8_t bestColR, bestColG, bestColB;
    uint32_t bestPos;

    /*__shared__ uint32_t bufferPos[18][16 + 2];
    __shared__ uint32_t bufferFgCol[18][16 + 2];*/

    //g.sync();
    //int* borderY = &(borderPos[fgBorderCount]);

    /*const int threadId = threadIdx.x * threadIdx.y;
    for (int i = threadId; i < fgBorderCount; i += 256) {
        borderPos[i] = fgBorderX[i];
        borderY[i] = fgBorderY[i];
    }*/

    if (trimap[id] != 128)
        return;

    // Propagation


    // Random search

    /*float minDist = FLT_MAX;
    for (int i = 0; i < fgBorderCount; i++) {
        float distance =
            x - fgBorderX[i]+
            y - fgBorderY[i]
        ;
        minDist = min(minDist, distance);
    }

    dists[id] = minDist;*/
}