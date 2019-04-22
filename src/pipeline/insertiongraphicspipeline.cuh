#ifndef INSERTION_GRAPHICS_PIPELINE
#define INSERTION_GRAPHICS_PIPELINE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/common.h"
#include "segmentation/vibe.cuh"
#include "matting/globalmatting.cuh"
#include "segmentation/shadow_detector.cuh"

class InsertionGraphicsPipeline
{
    private:
        ViBe* m_segmenter;
        ShadowDetector* m_shadowDectector;
        GlobalSampling* m_matting;

        // device buffers
        uint8_t* m_d_frame;
        uint8_t* m_d_segmentation;
        uint8_t* m_d_trimap;
        uint8_t* m_d_shadowIntensity;


        cv::cuda::GpuMat m_d_rgbFrame;
        cv::cuda::GpuMat m_d_rgbBg; 
        cv::cuda::GpuMat m_d_rgbGraphics;

        cv::cuda::GpuMat m_d_labFrame;
        cv::cuda::GpuMat m_d_labBg;
        cv::cuda::GpuMat m_d_labGraphics;
            
        // temp buffers
        uint8_t* m_d_temp_C4_UC;

    public:
        InsertionGraphicsPipeline();
        ~InsertionGraphicsPipeline();

    public:
        void initialize(Byte* frame);
        void process(Byte* input, Byte* graphics, Byte* output);
};


#endif