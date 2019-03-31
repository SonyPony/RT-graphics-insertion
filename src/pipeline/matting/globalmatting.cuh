#ifndef GLOBAL_MATTING
#define GLOBAL_MATTING

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"
#include "../../util/gpuutil.cuh"

using Gpu::Utils::RandState;

namespace Gpu {
    class GlobalSampling {
        private:
            static constexpr float COLOR_COST_COEFF = 10.f;

            const int m_width;
            const int m_height;
            const int m_size;

            RandState* m_d_randStates;

        public:
            GlobalSampling(int width, int height);

            void matting(Byte* d_image, Byte* d_trimap, Byte* d_output);
    };

}

#endif