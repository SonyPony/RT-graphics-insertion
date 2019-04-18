#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"
#include "../../common/config.h"
#include "../../util/gpuutil.cuh"

class GuidedFilter {
    private:
        int m_size;
        float* m_d_meanR;
        float* m_d_meanG;
        float* m_d_meanB;

        float* m_d_varRR;
        float* m_d_varRG;
        float* m_d_varRB;
        float* m_d_varGG;
        float* m_d_varGB;
        float* m_d_varBB;

        float* m_d_meanP;
        float* m_d_meanPR;
        float* m_d_meanPG;
        float* m_d_meanPB;

        float* m_d_AR;
        float* m_d_AG;
        float* m_d_AB;
        float* m_d_B;

    public:
        GuidedFilter();
        ~GuidedFilter();

        void filter(uchar4* frame, uint8_t* alphaMask, uint8_t* output);
};