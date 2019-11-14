#ifndef INSERTION_GRAPHICS_PIPELINE
#define INSERTION_GRAPHICS_PIPELINE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/common.h"
#include "segmentation/vibe.cuh"
#include "matting/globalmatting.cuh"
#include "segmentation/shadow_detector.cuh"

#include "../pipeline/trimap/trimap_generator.cuh"
#include "../pipeline/composer/composer.cuh"
#include "../pipeline/bginit/bghist.cuh"
#include <windows.h>
#include <cuda_gl_interop.h>


class InsertionGraphicsPipeline
{
    private:
        ViBe* m_segmenter;
        ShadowDetector* m_shadowDectector;
        TrimapGenerator* m_trimapGenerator;
        GlobalSampling* m_matting;
        Composer* m_composer;
        BgHist* m_bgHist;

        // device buffers
        uint8_t* m_d_frame;
        uint8_t* m_d_segmentation;
        uint8_t* m_d_trimap;
        uint8_t* m_d_shadowIntensity;
        uint8_t* m_d_graphicsAlphaMask;
        uint8_t* m_d_output;
        uint8_t* m_d_graphicsMask;

        uint8_t* m_d_transformedGraphics;

        cv::cuda::GpuMat m_d_rgbFrame;
        cv::cuda::GpuMat m_d_rgbBg; 
        cv::cuda::GpuMat m_d_rgbGraphics;

        cv::cuda::GpuMat m_d_labFrame;
        cv::cuda::GpuMat m_d_labBg;
        //cv::cuda::GpuMat m_d_labGraphics;
        
        cv::Mat m_transformMat;
        cv::Size m_graphicsSize;

        // temp buffers
        uint8_t* m_d_temp_C4_UC;
        uint8_t* m_d_temp2_C4_UC;

    public:
        InsertionGraphicsPipeline(/*cv::Size graphicsSize, cv::Point2f dstPoints[]*/);
        ~InsertionGraphicsPipeline();

    public:
        void initialize(Byte* frame);
        void process(Byte* input, Byte* graphics, Byte* output, Byte* bgOut = nullptr);

        void initAddFrame(Byte* frame);
        void computeInitBg(uint8_t* output);
        void initBgFromImage(uint8_t* input);

        void computeTransMatrix(cv::Size graphicsSize, cv::Point2f dstPoints[]);

        void createGraphicsResource(GLuint texId, cudaGraphicsMapFlags mapFlags);
        void deleteGraphicsResource();
};


#endif