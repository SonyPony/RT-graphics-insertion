#include "bghist.cuh"

BgHist::BgHist() {
    cudaMalloc(
        reinterpret_cast<void**>(&m_d_hist), 
        FRAME_SIZE * BgHist::BINS_COUNT * 3 * sizeof(float)
    ); // as 3 channels
    this->reset();
}

BgHist::~BgHist()
{
    cudaFree(m_d_hist);
}

__global__ void k_reset(float* hist) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    for (int i = 0; i < BgHist::BINS_COUNT; i++) {
        for (int c = 0; c < 3; c++) {
            hist[id + c * FRAME_SIZE + i * (3 * FRAME_SIZE)] = 0.f;
        }
    }
}

__global__ void k_addFrame(uchar4* frame, float* hist) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const int binStride = 256 / BgHist::BINS_COUNT;
    const uchar4 pixel = frame[id];

    atomicAdd(&hist[id + (pixel.x / binStride) * (3 * FRAME_SIZE)], 1.f);
    atomicAdd(&hist[id + FRAME_SIZE + (pixel.y / binStride) * (3 * FRAME_SIZE)], 1.f);
    atomicAdd(&hist[id + 2 * FRAME_SIZE + (pixel.z / binStride) * (3 * FRAME_SIZE)], 1.f);
}

__global__ void k_computeHist(float* hist, uchar4* dest) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const int binStride = 256 / BgHist::BINS_COUNT;
    uchar4 pixel {0, 0, 0, 255};
    float3 maxBins{0.f, 0.f, 0.f};

    for (int i = 0; i < BgHist::BINS_COUNT; i++) {
        int index = id + i * (3 * FRAME_SIZE);
        float R = hist[index];
        if (R > maxBins.x) {
            maxBins.x = R;
            pixel.x = i * binStride;
        }

        index = id + FRAME_SIZE + i * (3 * FRAME_SIZE);
        float G = hist[index];
        if (G > maxBins.y) {
            maxBins.y = G;
            pixel.y = i * binStride;
        }

        index = id + 2 * FRAME_SIZE + i * (3 * FRAME_SIZE);
        float B = hist[index];
        if (B > maxBins.z) {
            maxBins.z = B;
            pixel.z = i * binStride;
        }
    }

    dest[id] = pixel;
}

void BgHist::addFrame(uchar4 * d_frame)
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    k_addFrame << <dimGrid, dimBlock >> > (d_frame, m_d_hist);
}

void BgHist::reset()
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    k_reset << <dimGrid, dimBlock >> > (m_d_hist);
}

void BgHist::computeMode(uchar4 * d_dest)
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    k_computeHist << <dimGrid, dimBlock >> > (m_d_hist, d_dest);
}
