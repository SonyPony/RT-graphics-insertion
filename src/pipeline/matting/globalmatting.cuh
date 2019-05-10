#ifndef GLOBAL_MATTING
#define GLOBAL_MATTING

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"
#include "../../util/gpuutil.cuh"
#include "guidedfilter.cuh"

using Gpu::Utils::RandState;

struct __align__(16) UnknownPixel
{
    uint8_t bgR;
    uint8_t bgG;
    uint8_t bgB;

    uint8_t frameR;
    uint8_t frameG;
    uint8_t frameB;

    float bestCost;
    uint8_t currentAlpha;
    
    uint16_t x;
    uint16_t y;
    uint8_t _;
};

struct __align__(8) MattingSample
{
    uint8_t R;
    uint8_t G;
    uint8_t B;

    uint16_t x;
    uint16_t y;
    uint8_t _;
};


class GlobalSampling {
    private:
        static constexpr float COLOR_COST_COEFF = 10.f;

        RandState* m_d_randStates;
        GuidedFilter m_guidedFilter;

        UnknownPixel* m_d_unknownPixels;
        MattingSample* m_d_mattingSamples;
        int* m_d_bestSamplesIndexes;

        uint32_t* m_d_unknownPixelsCount;
        uint32_t* m_d_samplesCount;


        uint8_t* m_d_temp;

    public:
        GlobalSampling(uint8_t* d_tempBuffer);
        ~GlobalSampling();

        void matting(uchar4* d_frame, Byte* d_trimap, uchar4* d_background, Byte* d_output);
};


#endif