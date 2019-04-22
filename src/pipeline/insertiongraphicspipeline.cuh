#ifndef INSERTION_GRAPHICS_PIPELINE
#define INSERTION_GRAPHICS_PIPELINE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/common.h"
#include "segmentation/vibe.cuh"
#include "matting/globalmatting.cuh"


namespace Gpu {
    class InsertionGraphicsPipeline
    {
        private:
            ViBe* m_segmenter;
            GlobalSampling* m_matting;

            // device buffers
            Byte* m_d_frame;
            Byte* m_d_segmentation;
            Byte* m_d_trimap;

            // temp buffers
            uint8_t* m_d_temp_C4_UC;

        public:
            InsertionGraphicsPipeline();
            ~InsertionGraphicsPipeline();

        public:
            void initialize(Byte* frame);
            void process(Byte* input, Byte* graphics, Byte* output);
    };
}

#endif