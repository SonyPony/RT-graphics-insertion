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

__global__ void k_textureProp(
    uint8_t* labGraphics, uint8_t* labBg, 
    uint8_t* rgbGraphics, uint8_t* graphicsMask,
    float sumL, uint32_t* graphicsPixelsCount) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;
    const int pixelId = id * 3;
    
    if (graphicsMask[id] == 255) {
        const float avgL = sumL / static_cast<float>(*graphicsPixelsCount);
        const float3 pixel{ rgbGraphics[pixelId], rgbGraphics[pixelId + 1], rgbGraphics[pixelId + 2] };

        const float diffL = static_cast<float>(labBg[pixelId]) - avgL;
        const float ratio = (fabs(diffL) / 255.f) * GRAPHICS_OPACITY;
        const float aC = (diffL > 0) ? 255.f : 0.f;

        rgbGraphics[pixelId] = clamp(pixel.x + (aC - pixel.x) * ratio, 0.f, 255.f);
        rgbGraphics[pixelId + 1] = clamp(pixel.y + (aC - pixel.y) * ratio, 0.f, 255.f);
        rgbGraphics[pixelId + 2] = clamp(pixel.z + (aC - pixel.z) * ratio, 0.f, 255.f);

        /*labGraphics[id * 3] = clamp(
            static_cast<float>(labGraphics[id * 3])
            + (static_cast<float>(labBg[id * 3]) - avgL),
            0.f,
            255.f
        );*/
    }
}

__global__ void k_addShadows(uint8_t* rgbFrame, uint8_t* shadowIntensity, uint8_t* graphicsMask) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const int pixelId = id * 3;

    if (graphicsMask[id] == 255) {
        const float3 pixel{ rgbFrame[pixelId], rgbFrame[pixelId + 1], rgbFrame[pixelId + 2] };
        const float diffL = static_cast<float>(shadowIntensity[id]) * GRAPHICS_OPACITY;

        const float ratio = (fabs(diffL) / 255.f) * GRAPHICS_OPACITY;

        rgbFrame[pixelId] = clamp(pixel.x + pixel.x * ratio, 0.f, 255.f);
        rgbFrame[pixelId + 1] = clamp(pixel.y + pixel.y * ratio, 0.f, 255.f);
        rgbFrame[pixelId + 2] = clamp(pixel.z + pixel.z * ratio, 0.f, 255.f);

        /*labFrame[id * 3] = static_cast<uint8_t>(clamp(
            static_cast<float>(labFrame[id * 3])
            - static_cast<float>(shadowIntensity[id]) * GRAPHICS_OPACITY,
            0.f,
            254.f
        ));*/
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
    uint8_t * d_graphicsMask, uint8_t* d_graphicsAreaMask, uint8_t * d_dest)
{
    // texture propagation
    cudaMemset(reinterpret_cast<void*>(m_d_graphicsPixelsCount), 0, sizeof(uint32_t));
    k_LChannel << <DIM_GRID, DIM_BLOCK >> > (
        d_labBg, m_d_temp, d_graphicsAreaMask, m_d_graphicsPixelsCount
    );

    cv::Scalar sumL = cv::cuda::sum(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, m_d_temp),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, d_graphicsAreaMask)
    );

    const float u_sumL = sumL.val[0];

    cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_labGraphics),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::COLOR_Lab2RGB
    );

    k_textureProp << <DIM_GRID, DIM_BLOCK >> > (
        d_labGraphics, d_labBg, m_d_temp, d_graphicsMask, u_sumL, m_d_graphicsPixelsCount
    );

    /*cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_labGraphics),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::COLOR_Lab2RGB
    );*/

    // assembling
    k_asemble <<<DIM_GRID, DIM_BLOCK>> > (d_rgbFrame, d_alphaMask, m_d_temp, d_graphicsMask);

    // add shadows
    cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_rgbFrame),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::COLOR_RGB2Lab
    );

    /*m_blurFilter->apply(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, d_shadowIntensity),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, m_d_temp)
    );*/
    //k_addShadows << <dimGrid, dimBlock >> > (m_d_temp, d_shadowIntensity, d_graphicsMask);
    k_addShadows << <DIM_GRID, DIM_BLOCK >> > (d_rgbFrame, d_shadowIntensity, d_graphicsMask);
    
    cudaMemcpy(d_dest, d_rgbFrame, FRAME_SIZE * 3, cudaMemcpyDeviceToDevice);
    /*cv::cuda::cvtColor(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, m_d_temp),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC3, d_dest),
        cv::COLOR_Lab2RGB
    );*/
}
