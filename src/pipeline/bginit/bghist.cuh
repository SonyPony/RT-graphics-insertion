#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/config.h"


class BgHist {
    public:
        static constexpr int BINS_COUNT = 128;

    private:
        float* m_d_hist;

    public:
        BgHist();
        ~BgHist();

        void addFrame(uchar4* d_frame);
        void reset();
        void computeMode(uchar4* d_dest);
};