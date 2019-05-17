#include "insertiongraphicspipeline.cuh"
#include "../common/config.h"
#include "../pipeline/morphology/erosion.h"
#include "../pipeline/morphology/erosionFuncTemplate.h"
#include "../pipeline/segmentation/shadow_detector.cuh"
#include <QDebug>
#include "nppi_geometry_transforms.h"


void InsertionGraphicsPipeline::computeTransMatrix(cv::Size graphicsSize, cv::Point2f dstPoints[]) {
    m_graphicsSize = graphicsSize;
    cv::Point2f srcPoints[4];
    srcPoints[0] = cv::Point2f{ 0.f, 0.f };
    srcPoints[1] = cv::Point2f{ (float)graphicsSize.width, 0.f };
    srcPoints[3] = cv::Point2f{ 0.f, (float)graphicsSize.height };
    srcPoints[2] = cv::Point2f{ (float)graphicsSize.width, (float)graphicsSize.height };

    m_transformMat = cv::getPerspectiveTransform(srcPoints, dstPoints);
}

InsertionGraphicsPipeline::InsertionGraphicsPipeline() {
    cudaMalloc(reinterpret_cast<void**>(&m_d_temp_C4_UC), FRAME_SIZE * 4);  // single channel

    m_graphicsSize = cv::Size{};

    m_transformMat = cv::Mat::eye(3, 3, CV_32F);
    m_segmenter = new ViBe(m_d_temp_C4_UC);
    m_shadowDectector = new ShadowDetector;
    m_trimapGenerator = new TrimapGenerator;
    m_matting = new GlobalSampling(m_d_temp_C4_UC);
    m_composer = new Composer(m_d_temp_C4_UC);
    m_bgHist = new BgHist;

    // alloc buffers on device
    cudaMalloc(reinterpret_cast<void**>(&m_d_frame), FRAME_SIZE * Config::CHANNELS_COUNT_INPUT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_segmentation), FRAME_SIZE);    // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_trimap), FRAME_SIZE);  // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_shadowIntensity), FRAME_SIZE);  // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_graphicsAlphaMask), FRAME_SIZE);  // single channel
    cudaMalloc(reinterpret_cast<void**>(&m_d_output), FRAME_SIZE * Config::CHANNELS_COUNT_INPUT);
    cudaMalloc(reinterpret_cast<void**>(&m_d_transformedGraphics), FRAME_SIZE * Config::CHANNELS_COUNT_INPUT);

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
    cudaFree(m_d_graphicsAlphaMask);
    cudaFree(m_d_output);
    cudaFree(m_d_transformedGraphics);

    delete m_matting;
    delete m_shadowDectector;
    delete m_segmenter;
    delete m_trimapGenerator;
    delete m_composer;
    delete m_bgHist;
}

void InsertionGraphicsPipeline::initialize(Byte * frame)
{
    cudaMemcpy(m_d_temp_C4_UC, frame, FRAME_SIZE * sizeof(uchar4), cudaMemcpyHostToDevice);
    uchar4* d_bgInit = reinterpret_cast<uchar4*>(m_d_temp_C4_UC);
    m_segmenter->initialize(d_bgInit);
}

void InsertionGraphicsPipeline::process(Byte * input, Byte * graphics, Byte * output)
{
    cudaSetDevice(0);
    dim3 dimGrid{ 80, 45 };
    dim3 dimBlock{ 16, 16 };

    // copy data
    cudaMemcpy(m_d_frame, input, FRAME_SIZE * Config::CHANNELS_COUNT_INPUT, cudaMemcpyHostToDevice);
    cudaMemset(m_d_temp_C4_UC, 0, FRAME_SIZE * Config::CHANNELS_COUNT_INPUT);
    cudaMemcpy(
        m_d_temp_C4_UC, 
        graphics, 
        GRAPHICS_WIDTH * GRAPHICS_HEIGHT * Config::CHANNELS_COUNT_INPUT, 
        cudaMemcpyHostToDevice
    );
    
    // transform graphics
    cv::cuda::warpPerspective(
        cv::cuda::GpuMat{ m_graphicsSize, CV_8UC4, m_d_temp_C4_UC },
        cv::cuda::GpuMat{ cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, CV_8UC4, m_d_transformedGraphics },
        m_transformMat, cv::Size{ FRAME_WIDTH, FRAME_HEIGHT }, cv::INTER_NEAREST);
    
    uchar4* d_frame = reinterpret_cast<uchar4*>(m_d_frame);
    uchar4* d_graphics = reinterpret_cast<uchar4*>(m_d_transformedGraphics);

    // segmentation
    uchar4* d_background = m_segmenter->segment(d_frame, m_d_segmentation);

    // split alpha channel
    Gpu::Utils::dualCvtRGBA2RGB(
        dimGrid, dimBlock,
        d_frame, d_background,
        m_d_rgbFrame.ptr(), m_d_rgbBg.ptr());
    Gpu::Utils::cvtRGBA2RGB_A(
        dimGrid, dimBlock, d_graphics, m_d_rgbGraphics.ptr(), m_d_graphicsAlphaMask
    );

    // convert to LAB
    cv::cuda::cvtColor(m_d_rgbBg, m_d_labBg, cv::COLOR_RGB2Lab);
    cv::cuda::cvtColor(m_d_rgbFrame, m_d_labFrame, cv::COLOR_RGB2Lab);
    cv::cuda::cvtColor(m_d_rgbGraphics, m_d_labGraphics, cv::COLOR_RGB2Lab);
        
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

    // assemble
    m_composer->compose(
        m_d_segmentation, m_d_shadowIntensity,
        m_d_rgbFrame.ptr(), m_d_labFrame.ptr(), m_d_labGraphics.ptr(), m_d_labBg.ptr(),
        m_d_graphicsAlphaMask, m_d_output
    );

    cudaMemcpy(output, m_d_output, FRAME_SIZE * 3, cudaMemcpyDeviceToHost);
}

void InsertionGraphicsPipeline::initAddFrame(Byte * frame)
{
    cudaMemcpy(m_d_frame, frame, FRAME_SIZE * Config::CHANNELS_COUNT_INPUT, cudaMemcpyHostToDevice);
    uchar4* d_frame = reinterpret_cast<uchar4*>(m_d_frame);

    m_bgHist->addFrame(d_frame);
}

void InsertionGraphicsPipeline::computeInitBg(uint8_t* output)
{
    uchar4* d_bgInit = reinterpret_cast<uchar4*>(m_d_temp_C4_UC);
    m_bgHist->computeMode(d_bgInit);
    m_segmenter->initialize(d_bgInit);
    m_bgHist->reset();
    cudaMemcpy(output, m_d_temp_C4_UC, FRAME_SIZE * 4, cudaMemcpyDeviceToHost);
}
