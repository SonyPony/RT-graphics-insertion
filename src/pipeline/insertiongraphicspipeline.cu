#include "insertiongraphicspipeline.cuh"
#include "../common/config.h"
#include "../pipeline/morphology/erosion.h"
#include "../pipeline/morphology/erosionFuncTemplate.h"


Gpu::InsertionGraphicsPipeline::InsertionGraphicsPipeline(int width, int height) 
    : m_width{ width }, m_height{ height }, m_size{ height * width } {
    m_segmenter = new ViBe(width, height);
    m_matting = new GlobalSampling(width, height);

    // alloc buffers on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_frame), m_size * Config::CHANNELS_COUNT_INPUT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_segmentation), m_size);    // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_trimap), m_size);  // single channel

}

Gpu::InsertionGraphicsPipeline::~InsertionGraphicsPipeline()
{
    cudaFree(m_d_frame);
    cudaFree(m_d_segmentation);
    delete m_segmenter;
}

void Gpu::InsertionGraphicsPipeline::initialize(Byte * frame)
{
    m_segmenter->initialize(frame);
}

void Gpu::InsertionGraphicsPipeline::process(Byte * input, Byte * graphics, Byte * output, /*temp*/ Byte* trimap)
{
    cudaSetDevice(0);

    // segmentation
    cudaMemcpy(m_d_frame, input, m_size * Config::CHANNELS_COUNT_INPUT, cudaMemcpyHostToDevice);
    Byte* d_background = m_segmenter->segment(m_d_frame, m_d_segmentation);
    cudaMemcpy(output, m_d_segmentation, m_size, cudaMemcpyDeviceToHost);


    uint8_t* d_temp;
    uint8_t* d_temp2;
    cudaMalloc(reinterpret_cast<void**>(&d_temp), m_size);
    cudaMalloc(reinterpret_cast<void**>(&d_temp2), m_size);
    cudaMemcpy(d_temp2, m_d_segmentation, m_size, cudaMemcpyDeviceToDevice);

    
    ErosionTemplateSharedTwoSteps(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 2);
    FilterDilation(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 2);

    
    FilterDilation(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 1);
    ErosionTemplateSharedTwoSteps(m_d_segmentation, d_temp, FRAME_WIDTH, FRAME_HEIGHT, 1);
    
    // trimap

    // matting
    cudaMemcpy(m_d_trimap, trimap, m_size, cudaMemcpyHostToDevice);
    m_matting->matting(m_d_frame, m_d_trimap, d_background, m_d_segmentation);

    // TEST output
    cudaMemcpy(output, m_d_segmentation, m_size, cudaMemcpyDeviceToHost);
}
