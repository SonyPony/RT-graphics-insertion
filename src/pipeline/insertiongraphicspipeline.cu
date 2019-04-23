#include "insertiongraphicspipeline.cuh"
#include "../common/config.h"
#include "../pipeline/morphology/erosion.h"
#include "../pipeline/morphology/erosionFuncTemplate.h"
#include "../pipeline/segmentation/shadow_detector.cuh"


InsertionGraphicsPipeline::InsertionGraphicsPipeline() {
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp_C4_UC), FRAME_SIZE * 4);  // single channel

    m_segmenter = new ViBe(m_d_temp_C4_UC);
    m_shadowDectector = new ShadowDetector;
    m_trimapGenerator = new TrimapGenerator;
    m_matting = new GlobalSampling(m_d_temp_C4_UC);


    // alloc buffers on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_frame), FRAME_SIZE * Config::CHANNELS_COUNT_INPUT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_segmentation), FRAME_SIZE);    // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_trimap), FRAME_SIZE);  // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_shadowIntensity), FRAME_SIZE);  // single channel


    m_d_rgbBg = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_rgbFrame = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_rgbGraphics = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_labBg = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_labFrame = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
    m_d_labGraphics = cv::cuda::createContinuous(FRAME_WIDTH, FRAME_HEIGHT, CV_8UC3);
}

InsertionGraphicsPipeline::~InsertionGraphicsPipeline()
{
    cudaFree(m_d_frame);
    cudaFree(m_d_segmentation);
    cudaFree(m_d_temp_C4_UC);
    cudaFree(m_d_trimap);
    cudaFree(m_d_shadowIntensity);

    delete m_matting;
    delete m_shadowDectector;
    delete m_segmenter;
    delete m_trimapGenerator;
}

void InsertionGraphicsPipeline::initialize(Byte * frame)
{
    m_segmenter->initialize(frame);
}

void InsertionGraphicsPipeline::process(Byte * input, Byte * graphics, Byte * output)
{
    cudaSetDevice(0);
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    // copy data
    cudaMemcpy(m_d_frame, input, FRAME_SIZE * Config::CHANNELS_COUNT_INPUT, cudaMemcpyHostToDevice);
    uchar4* d_frame = reinterpret_cast<uchar4*>(m_d_frame);

    // segmentation
    uchar4* d_background = m_segmenter->segment(d_frame, m_d_segmentation);

    // convert to LAB
    Gpu::Utils::dualCvtRGBA2RGB(
        dimGrid, dimBlock, 
        d_frame, d_background,
        m_d_rgbFrame.ptr(), m_d_rgbBg.ptr());
    cv::cuda::cvtColor(m_d_rgbBg, m_d_labBg, cv::COLOR_RGB2Lab);
    cv::cuda::cvtColor(m_d_rgbFrame, m_d_labFrame, cv::COLOR_RGB2Lab);
    // TODO graphics
    
    // shadow segmentation
    m_shadowDectector->process(d_frame, m_d_segmentation, d_background, 
        m_d_labFrame.ptr(), m_d_labBg.ptr(), m_d_shadowIntensity);

    // mophology refinement
    ErosionTemplateSharedTwoSteps(m_d_segmentation, m_d_temp_C4_UC, FRAME_WIDTH, FRAME_HEIGHT, 2);
    FilterDilation(m_d_segmentation, m_d_temp_C4_UC, FRAME_WIDTH, FRAME_HEIGHT, 2);

    FilterDilation(m_d_segmentation, m_d_temp_C4_UC, FRAME_WIDTH, FRAME_HEIGHT, 1);
    ErosionTemplateSharedTwoSteps(m_d_segmentation, m_d_temp_C4_UC, FRAME_WIDTH, FRAME_HEIGHT, 1);

    // trimap generation
    m_trimapGenerator->generate(m_d_segmentation, m_d_trimap);

    // image matting
    m_matting->matting(d_frame, m_d_trimap, d_background, m_d_segmentation);



    // TEST output
    cudaMemcpy(output, m_d_segmentation, FRAME_SIZE, cudaMemcpyDeviceToHost);
}
