#include "shadow_detector.cuh"
#include "../../util/gpuutil.cuh"
#include "helper_math.h"
#include "../../common/common.h"

ShadowDetector::ShadowDetector() {
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp16), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));
    cudaMalloc(reinterpret_cast<void**>(&m_d_gradsFrame), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));
    cudaMalloc(reinterpret_cast<void**>(&m_d_gradsBg), FRAME_WIDTH * FRAME_HEIGHT * sizeof(short2));

    cudaMalloc(reinterpret_cast<void**>(&m_d_grayFrame), FRAME_WIDTH * FRAME_HEIGHT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_grayBg), FRAME_WIDTH * FRAME_HEIGHT);
}

ShadowDetector::~ShadowDetector()
{
    cudaFree(m_d_temp16);
    cudaFree(m_d_gradsFrame);
    cudaFree(m_d_gradsBg);
    cudaFree(m_d_gradsFrame);
    cudaFree(m_d_grayBg);
}

__global__ void k_grayscaleInputs(uchar4* frame, uchar4* bgModel, uint8_t* outFrame, uint8_t* outBgModel) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;

    const uchar4 bgPixel = bgModel[id];
    const uchar4 framePixel = frame[id];

    outFrame[id] = Gpu::Utils::cvtRGB2GRAY(framePixel);
    outBgModel[id] = Gpu::Utils::cvtRGB2GRAY(bgPixel);
}

__global__ void k_segmentShadow(short2* frameGrads, 
    short2* bgGrads, uint8_t* segmentationMask, uint8_t* shadow, uint8_t* labFrame, uint8_t* labBg) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int id = x + y * FRAME_WIDTH;
    
    const uchar3 labFramePixel = {labFrame[id * 3], labFrame[id * 3 + 1] , labFrame[id * 3 + 2] };
    const uchar3 labBgPixel = { labBg[id * 3], labBg[id * 3 + 1], labBg[id * 3 + 2] };
   
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

    uint8_t shadowVal = 0;
    if (colorDist < ShadowDetector::COLOR_RADIUS 
        && segmentationMask[id] == FOREGROUND
        && labFramePixel.x >= ShadowDetector::LIGHTNESS_THRESHOLD
        && labBgPixel.x > labFramePixel.x
        && textureDist < ShadowDetector::TEXTURE_RADIUS
    ) {
        segmentationMask[id] = 0;
        shadowVal = labBgPixel.x - labFramePixel.x;
    }
    shadow[id] = shadowVal;
}

void ShadowDetector::process(uchar4* frame, uint8_t * d_segmentationMask, uchar4* d_bgModel, 
    uint8_t* labFrame, uint8_t* labBg, uint8_t * dest)
{
    k_grayscaleInputs << <DIM_GRID, DIM_BLOCK >> > (frame, d_bgModel, m_d_grayFrame,
        m_d_grayBg
    );

    Gpu::Utils::gradients(DIM_GRID, DIM_BLOCK, m_d_grayFrame, m_d_temp16, m_d_gradsFrame);
    Gpu::Utils::gradients(DIM_GRID, DIM_BLOCK, m_d_grayBg, m_d_temp16, m_d_gradsBg);

    k_segmentShadow << <DIM_GRID, DIM_BLOCK >> > (m_d_gradsFrame, m_d_gradsBg, 
        d_segmentationMask, dest, labFrame, labBg);
}

