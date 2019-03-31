#ifndef VIBE
#define VIBE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"
#include "../../util/gpuutil.cuh"

using Gpu::Utils::RandState;

namespace Gpu {
    class ViBe {
        private:
            static constexpr uint8_t SAMPLE_COUNT = 20;

            Byte* m_d_bgModel;
            RandState* m_d_randState;

            const int m_width;
            const int m_height;
            const int m_size;

        public:
            ViBe(int width, int height);
            ~ViBe();

            void initialize(Byte* backgroundModel);
            void segment(Byte* d_input, Byte* d_dest);
    };
    
}

#endif // !VIBE
