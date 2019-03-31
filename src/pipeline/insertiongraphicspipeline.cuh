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
            const int m_width;
            const int m_height;
            const int m_size;

            ViBe* m_segmenter;
            GlobalSampling* m_matting;

            // device buffers
            Byte* m_d_frame;
            Byte* m_d_segmentation;
            Byte* m_d_trimap;

        public:
            InsertionGraphicsPipeline(int width, int height);
            ~InsertionGraphicsPipeline();

        public:
            void initialize(Byte* frame);
            void process(Byte* input, Byte* graphics, Byte* output, /*temp*/ Byte* trimap);
    };
}

#endif