#include "composer.cuh"


Composer::Composer(uint8_t* d_tempBuffer) {
    m_blurFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size{ 5, 5 }, 5);
    m_d_temp = d_tempBuffer;

    m_d_matBuffer = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC4);
    //m_d_rgbGraphics = cudaMalloc()
    cudaMalloc(reinterpret_cast<void**>(&m_d_graphicsPixelsCount), sizeof(uint32_t));
}

Composer::~Composer()
{
    cudaFree(m_d_graphicsPixelsCount);
}

__global__ void k_LChannel(
    uint8_t* in, uint8_t* out, uint8_t* graphicsMask, uint32_t* graphicsPixelsCount) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    out[id] = in[id * 3];
    if(graphicsMask[id] == 255)
        atomicAdd(graphicsPixelsCount, 1);
}

__global__ void k_textureProp(uint8_t* labGraphics, uint8_t* labBg, uint8_t* graphicsMask, 
    float sumL, uint32_t* graphicsPixelsCount) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const float avgL = sumL / static_cast<float>(*graphicsPixelsCount);

    labGraphics[id * 3] = clamp(
        static_cast<float>(labGraphics[id * 3])
        + (static_cast<float>(labBg[id * 3]) - avgL),
        0.f,
        255.f
    );
}

__global__ void k_addShadows(uint8_t* labFrame, uint8_t* shadowIntensity, uint8_t* graphicsMask) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    if (graphicsMask[id] == 255) {
        labFrame[id * 3] = clamp(
            static_cast<float>(labFrame[id * 3])
            - static_cast<float>(shadowIntensity[id]) * GRAPHICS_OPACITY,
            0.f,
            255.f
        );
    }
}

__global__ void k_asemble(uint8_t* rgbFrame, uint8_t* foregroundMask, uint8_t* rgbGraphics, uint8_t* graphicsMask) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;
    const int pixelId = id * 3;     // stride 3, beacause of mutichannel

    const float3 framePixel = { rgbFrame[pixelId], rgbFrame[pixelId + 1], rgbFrame[pixelId + 2] };
    const float3 graphicsPixels = { rgbGraphics[pixelId] , rgbGraphics[pixelId + 1], rgbGraphics[pixelId + 2] };
    const float alpha = static_cast<float>(foregroundMask[id]) / 255.f;

    float3 resultPixel = framePixel;

    if (graphicsMask[id] == 255) {
        resultPixel = clamp(
            (1.f - GRAPHICS_OPACITY) * resultPixel  + GRAPHICS_OPACITY * graphicsPixels,
            0.f, 255.f
        );

        resultPixel = clamp(
            alpha * framePixel + (1.f - alpha) * resultPixel,
            0.f, 255.f
        );
    }

    rgbFrame[pixelId] = resultPixel.x;
    rgbFrame[pixelId + 1] = resultPixel.y;
    rgbFrame[pixelId + 2] = resultPixel.z;
}

void Composer::compose(uint8_t * d_alphaMask, uint8_t * d_shadowIntensity, 
    uint8_t * d_rgbFrame, uint8_t * d_labFrame, uint8_t * d_labGraphics, uint8_t * d_labBg,
    uint8_t * d_graphicsMask, uint8_t * d_dest)
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    // texture propagation
    cudaMemset(reinterpret_cast<void*>(m_d_graphicsPixelsCount), 0, sizeof(uint32_t));
    k_LChannel << <dimGrid, dimBlock >> > (
        d_labBg, m_d_temp, d_graphicsMask, m_d_graphicsPixelsCount
    );

    cv::Scalar sumL = cv::cuda::sum(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, m_d_temp),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, d_graphicsMask)
    );

    const float u_sumL = sumL.val[0];
    k_textureProp << <dimGrid, dimBlock >> > (
        d_labGraphics, d_labBg, d_graphicsMask, u_sumL, m_d_graphicsPixelsCount
    );

    cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_labGraphics),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::COLOR_Lab2RGB
    );

    // assembling
    k_asemble << <dimGrid, dimBlock >> > (d_rgbFrame, d_alphaMask, m_d_temp, d_graphicsMask);
    
    // add shadows
    cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_rgbFrame),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::COLOR_RGB2Lab
    );

    m_blurFilter->apply(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, d_shadowIntensity),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, m_d_temp)
    );
    /*k_addShadows << <dimGrid, dimBlock >> > (m_d_temp, d_shadowIntensity, d_graphicsMask);

    cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_dest),
        cv::COLOR_Lab2RGB
    );
}
