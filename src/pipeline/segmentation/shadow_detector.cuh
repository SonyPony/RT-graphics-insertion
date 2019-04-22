#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/config.h"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/imgproc.hpp"


class ShadowDetector {
    public:
        static constexpr float COLOR_RADIUS = 10.f;
        static constexpr uint8_t LIGHTNESS_THRESHOLD = 40;
        static constexpr float TEXTURE_RADIUS = 60.f;

    private:
        short2* m_d_temp16;

        short2* m_d_gradsBg;
        short2* m_d_gradsFrame;

        uint8_t* m_d_grayBg;
        uint8_t* m_d_grayFrame;

        /*cv::cuda::GpuMat m_d_bgrFrame;
        cv::cuda::GpuMat m_d_bgrBg;

        cv::cuda::GpuMat m_d_labFrame;
        cv::cuda::GpuMat m_d_labBg;
        */
    public:
        ShadowDetector();
        ~ShadowDetector();

        void process(uchar4* frame, uint8_t* d_segmentationMask, uchar4* d_bgModel, 
            uint8_t* labFrame, uint8_t* labBg, uint8_t* dest);
};