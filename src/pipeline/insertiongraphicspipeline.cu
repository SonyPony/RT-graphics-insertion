#include "insertiongraphicspipeline.cuh"
#include "../common/config.h"


Gpu::InsertionGraphicsPipeline::InsertionGraphicsPipeline(int width, int height) 
    : m_width{ width }, m_height{ height }, m_size{ height * width } {
    m_segmenter = new ViBe(width, height);

    // alloc buffers on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_frame), m_size * Config::CHANNELS_COUNT_INPUT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_segmentation), m_size);    // single channel

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

void Gpu::InsertionGraphicsPipeline::process(Byte * input, Byte * graphics, Byte * output)
{
    cudaSetDevice(0);

    // segmentation
    cudaMemcpy(m_d_frame, input, m_size * Config::CHANNELS_COUNT_INPUT, cudaMemcpyHostToDevice);
    m_segmenter->segment(m_d_frame, m_d_segmentation);
    cudaMemcpy(output, m_d_segmentation, m_size, cudaMemcpyDeviceToHost);

    // matting
}
