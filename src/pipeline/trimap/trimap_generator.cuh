#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/config.h"


class TrimapGenerator {
    private:
        short2* m_d_temp16;
        short2* m_d_grads;
        uint8_t* m_d_temp;
        uint8_t* m_d_dilatatedEdges;

    public:
        TrimapGenerator();
        ~TrimapGenerator();

        void generate(uint8_t* d_segmentationMask, uint8_t* d_dest);
};


