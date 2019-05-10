#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"
#include "../../common/config.h"
#include "../../util/gpuutil.cuh"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"


class Composer {
    private:
        cv::Ptr<cv::cuda::Filter> m_blurFilter;
        cv::cuda::GpuMat m_d_matBuffer;
        uint8_t* m_d_temp;
        uint32_t* m_d_graphicsPixelsCount;

    public:
        Composer(uint8_t* d_tempBuffer);
        ~Composer();

        void compose(
            uint8_t* d_alphaMask,
            uint8_t* d_shadowIntensity,

            uint8_t * d_rgbFrame,
            uint8_t* d_labFrame,
            uint8_t* d_labGraphics,
            uint8_t* d_labBg,
        
            uint8_t* d_graphicsMask,

            uint8_t* d_dest
        );
};