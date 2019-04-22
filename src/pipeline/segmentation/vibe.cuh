#ifndef VIBE
#define VIBE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"
#include "../../util/gpuutil.cuh"

using Gpu::Utils::RandState;

namespace Gpu {
    class ViBe {
        public:
            static constexpr uint8_t SAMPLE_COUNT = 20;
            static constexpr uint8_t COLOR_RADIUS = 20;

        private:
            uchar4* m_d_bgModel;
            RandState* m_d_randState;
            
            uint8_t* m_d_temp;

        public:
            ViBe(uint8_t* d_tempBuffer);
            ~ViBe();

            void initialize(uint8_t* backgroundModel);
            uchar4* segment(uchar4* d_input, uint8_t* d_dest);
    };
    
}

#endif // !VIBE
