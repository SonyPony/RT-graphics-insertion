#include "composer.cuh"
#include <QDebug>


Composer::Composer(uint8_t* d_tempBuffer) {
    m_blurFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size{ 5, 5 }, 5);
    m_d_temp = d_tempBuffer;

    m_d_matBuffer = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC4);
    //m_d_rgbGraphics = cudaMalloc()
    cudaMalloc(reinterpret_cast<void**>(&m_d_graphicsPixelsCount), sizeof(uint32_t));

    m_d_sum = cv::cuda::createContinuous(1, 1, CV_32F);
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
    uint8_t* labBg, uint8_t* rgbGraphics, uint8_t* graphicsMask,
    double* sumL, uint32_t* graphicsPixelsCount) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;
    const int pixelId = id * 3;
    
    if (graphicsMask[id] != 0) {
        const float avgL = sumL[0] / static_cast<float>(*graphicsPixelsCount);
        const float3 pixel{ rgbGraphics[pixelId], rgbGraphics[pixelId + 1], rgbGraphics[pixelId + 2] };

        const float diffL = static_cast<float>(labBg[pixelId]) - avgL;
        const float ratio = (fabs(diffL) / (255.f / TEXTURE_ENHANCE_LEVEL));
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

__global__ void k_addShadows(
    uint8_t* rgbFrame, uint8_t* shadowIntensity, 
    uint8_t* graphicsMask, uint8_t* dest) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const int pixelId = id * 3;

    float3 pixel{ rgbFrame[pixelId], rgbFrame[pixelId + 1], rgbFrame[pixelId + 2] };

    if (graphicsMask[id] != 0) {
        const float diffL = static_cast<float>(shadowIntensity[id]);

        const float ratio = (fabs(diffL) / 255.f);

        pixel.x = clamp(pixel.x + pixel.x * ratio, 0.f, 255.f);
        pixel.y = clamp(pixel.y + pixel.y * ratio, 0.f, 255.f);
        pixel.z = clamp(pixel.z + pixel.z * ratio, 0.f, 255.f);

        /*labFrame[id * 3] = static_cast<uint8_t>(clamp(
            static_cast<float>(labFrame[id * 3])
            - static_cast<float>(shadowIntensity[id]) * GRAPHICS_OPACITY,
            0.f,
            254.f
        ));*/
    }

    dest[pixelId] = pixel.x;
    dest[pixelId + 1] = pixel.y;
    dest[pixelId + 2] = pixel.z;
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
    const float graphicsAlpha = graphicsMask[id];

    if (graphicsAlpha != 0) {
        const float graphicsOpacity = (graphicsAlpha / 255.f);

        resultPixel = clamp(
            (1.f - graphicsOpacity) * resultPixel  + graphicsOpacity * graphicsPixels,
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
    uint8_t * d_rgbFrame, uint8_t * d_labFrame, uint8_t * d_rgbGraphics, uint8_t * d_labBg,
    uint8_t * d_graphicsMask, uint8_t* d_graphicsAreaMask, uint8_t * d_dest)
{
    // texture propagation
    cudaMemset(reinterpret_cast<void*>(m_d_graphicsPixelsCount), 0, sizeof(uint32_t));
    k_LChannel << <DIM_GRID, DIM_BLOCK >> > (
        d_labBg, m_d_temp, d_graphicsAreaMask, m_d_graphicsPixelsCount
    );

    /*cv::Scalar sumL = cv::cuda::sum(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, m_d_temp),
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, d_graphicsAreaMask)
    );

    const float u_sumL = sumL.val[0];*/

    cv::cuda::calcSum(
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, m_d_temp),
        m_d_sum,
        cv::cuda::GpuMat(cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC1, d_graphicsAreaMask)
    );

    k_textureProp << <DIM_GRID, DIM_BLOCK >> > (
        d_labBg, d_rgbGraphics, d_graphicsMask, m_d_sum.ptr<double>(), m_d_graphicsPixelsCount
    );

    // assembling
    k_asemble <<<DIM_GRID, DIM_BLOCK>> > (d_rgbFrame, d_alphaMask, d_rgbGraphics, d_graphicsMask);

    // add shadows
    k_addShadows << <DIM_GRID, DIM_BLOCK >> > (d_rgbFrame, d_shadowIntensity, d_graphicsMask, d_dest);
}
