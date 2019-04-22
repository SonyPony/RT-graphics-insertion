#include "insertiongraphicspipeline.cuh"
#include "../common/config.h"
#include "../pipeline/morphology/erosion.h"
#include "../pipeline/morphology/erosionFuncTemplate.h"
#include "../pipeline/trimap/trimap_generator.cuh"
#include "../pipeline/segmentation/shadow_detector.cuh"

#include <npp.h>

Gpu::InsertionGraphicsPipeline::InsertionGraphicsPipeline() {
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp_C4_UC), FRAME_SIZE * 4);  // single channel

    m_segmenter = new ViBe(m_d_temp_C4_UC);
    m_matting = new GlobalSampling;

    // alloc buffers on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_frame), FRAME_SIZE * Config::CHANNELS_COUNT_INPUT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_segmentation), FRAME_SIZE);    // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_trimap), FRAME_SIZE);  // single channel
    

}

Gpu::InsertionGraphicsPipeline::~InsertionGraphicsPipeline()
{
    cudaFree(m_d_frame);
    cudaFree(m_d_segmentation);
    cudaFree(m_d_temp_C4_UC);
    cudaFree(m_d_trimap);

    delete m_matting;
    delete m_segmenter;
}

void Gpu::InsertionGraphicsPipeline::initialize(Byte * frame)
{
    m_segmenter->initialize(frame);
}

void Gpu::InsertionGraphicsPipeline::process(Byte * input, Byte * graphics, Byte * output)
{
    cudaSetDevice(0);

    uchar4* d_frame = reinterpret_cast<uchar4*>(m_d_frame);

    // segmentation
    cudaMemcpy(m_d_frame, input, FRAME_SIZE * Config::CHANNELS_COUNT_INPUT, cudaMemcpyHostToDevice);
    /*Byte* d_background = */m_segmenter->segment(d_frame, m_d_segmentation);
    cudaMemcpy(output, m_d_segmentation, FRAME_SIZE, cudaMemcpyDeviceToHost);


    uint8_t* d_shadowIntensity;
    cudaMalloc(reinterpret_cast<void**>(&d_shadowIntensity), FRAME_SIZE);
    uint8_t* d_trimap;
    cudaMalloc(reinterpret_cast<void**>(&d_trimap), FRAME_SIZE);

    uint8_t* d_temp;
    uint8_t* d_temp2;
    cudaMalloc(reinterpret_cast<void**>(&d_temp), FRAME_SIZE);
    cudaMalloc(reinterpret_cast<void**>(&d_temp2), FRAME_SIZE);
    cudaMemcpy(d_temp2, m_d_segmentation, FRAME_SIZE, cudaMemcpyDeviceToDevice);

    
    
    
    // trimap
    /*TrimapGenerator trimapGen;
    cudaMemcpy(d_temp, m_d_segmentation, m_size, cudaMemcpyDeviceToDevice);
    trimapGen.generate(d_temp, m_d_segmentation);*/

    // matting
    //cudaMemcpy(m_d_trimap, trimap, FRAME_SIZE, cudaMemcpyHostToDevice);
    //m_matting->matting(m_d_frame, m_d_trimap, d_background, m_d_segmentation);

    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    uchar4* d_frame = reinterpret_cast<uchar4*>(m_d_frame);
    //cudaMemcpy(m_d_segmentation, d_temp2, m_size, cudaMemcpyDeviceToDevice);
    ShadowDetector shadowDectector;
    shadowDectector.process(d_frame, m_d_segmentation, d_background, d_shadowIntensity);

    
    ErosionTemplateSharedTwoSteps(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 2);
    FilterDilation(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 2);


    FilterDilation(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 1);
    ErosionTemplateSharedTwoSteps(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 1);

    TrimapGenerator trimapGen;
    trimapGen.generate(m_d_segmentation, d_trimap);

    m_matting->matting(m_d_frame, d_trimap, d_background, m_d_segmentation);

    // TEST output
    cudaMemcpy(output, m_d_segmentation, FRAME_SIZE, cudaMemcpyDeviceToHost);
}
