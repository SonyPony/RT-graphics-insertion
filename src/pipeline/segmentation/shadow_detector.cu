#include "shadow_detector.cuh"
#include "../../util/gpuutil.cuh"
#include "helper_math.h"
#include <npp.h>

ShadowDetector::ShadowDetector() {
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp16), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));
    cudaMalloc(reinterpret_cast<void**>(&m_d_gradsFrame), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));
    cudaMalloc(reinterpret_cast<void**>(&m_d_gradsBg), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));

    cudaMalloc(reinterpret_cast<void**>(&m_d_grayFrame), FRAME_WIDTH * FRAME_HEIGHT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_grayBg), FRAME_WIDTH * FRAME_HEIGHT);

    m_d_bgrBg = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_bgrFrame = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_labBg = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_labFrame = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
}

ShadowDetector::~ShadowDetector()
{
    cudaFree(m_d_temp16);
    cudaFree(m_d_gradsFrame);
    cudaFree(m_d_gradsBg);
    cudaFree(m_d_gradsFrame);
    cudaFree(m_d_grayBg);
}

__global__ void k_prepareInputs(uchar4* frame, uint8_t* bgModel, uint8_t* outFrame, uint8_t* outBgModel,
    uint8_t* bgrBg, uint8_t* bgrFrame) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    uchar4 bgPixel;
    bgPixel.x = bgModel[id];
    bgPixel.y = bgModel[id + FRAME_WIDTH * FRAME_HEIGHT];
    bgPixel.z = bgModel[id + 2 * FRAME_WIDTH * FRAME_HEIGHT];

    const uchar4 framePixel = frame[id];

    outFrame[id] = Gpu::Utils::cvtRGB2GRAY(framePixel);
    outBgModel[id] = Gpu::Utils::cvtRGB2GRAY(bgPixel);

    bgrBg[id * 3] = bgPixel.z;
    bgrBg[id * 3 + 1] = bgPixel.y;
    bgrBg[id * 3 + 2] = bgPixel.x;

    bgrFrame[id * 3] = framePixel.z;
    bgrFrame[id * 3 + 1] = framePixel.y;
    bgrFrame[id * 3 + 2] = framePixel.x;
}

__global__ void k_segmentShadow(short2* frameGrads, 
    short2* bgGrads, uint8_t* segmentationMask, uint8_t* shadow, uint8_t* labFrame, uint8_t* labBg) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;
    
    uchar3 labFramePixel = {labFrame[id * 3], labFrame[id * 3 + 1] , labFrame[id * 3 + 2] };// = Gpu::Utils::cvtRGB2LAB(framePixel);
    uchar3 labBgPixel = { labBg[id * 3], labBg[id * 3 + 1], labBg[id * 3 + 2] };// = Gpu::Utils::cvtRGB2LAB(bgPixel);
    

    const short2 frameGradient = frameGrads[id];
    const short2 bgGradient = bgGrads[id];

    const float colorDist = norm3df(
        (float)labFramePixel.z - labBgPixel.z,
        (float)labFramePixel.y - labBgPixel.y,
        0.f
    );

    const float textureDist = norm3df(
        (float)frameGradient.x - bgGradient.x,
        (float)frameGradient.y - bgGradient.y,
        0.f
    );

    if (colorDist < 10 
        && segmentationMask[id] == 255
        && labFramePixel.x >= 40
        && labBgPixel.x > labFramePixel.x
        && textureDist < 60) {
        segmentationMask[id] = 0;
        shadow[id] = labBgPixel.x - labFramePixel.x;
    }
}

void ShadowDetector::process(uchar4* frame, uint8_t * d_segmentationMask, uint8_t * d_bgModel, uint8_t * dest)
{
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };


    k_prepareInputs << <dimGrid, dimBlock >> > (frame, d_bgModel, m_d_grayFrame, 
        m_d_grayBg, m_d_bgrBg.ptr(), m_d_bgrFrame.ptr()
    );
    Gpu::Utils::gradients(dimGrid, dimBlock, m_d_grayFrame, m_d_temp16, m_d_gradsFrame);
    Gpu::Utils::gradients(dimGrid, dimBlock, m_d_grayBg, m_d_temp16, m_d_gradsBg);

    cv::cuda::cvtColor(m_d_bgrBg, m_d_labBg, cv::COLOR_BGR2Lab);
    cv::cuda::cvtColor(m_d_bgrFrame, m_d_labFrame, cv::COLOR_BGR2Lab);

    k_segmentShadow << <dimGrid, dimBlock >> > (m_d_gradsFrame, m_d_gradsBg, 
        d_segmentationMask, dest, m_d_labFrame.ptr(), m_d_labBg.ptr());
}

