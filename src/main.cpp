#include "pipeline/insertiongraphicspipeline.cuh"
#include <QtWidgets/QApplication>
#include "pipeline/segmentation/vibe.cuh"
#include <iostream>
#include <QLabel>
#include <QDebug>
#include "util/util.h"
#include <time.h>

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "opencv2/imgproc.hpp"
#include <chrono>

using namespace std;

using namespace cv;

int main(int argc, char *argv[])
{
   /* cudaSetDevice(0);
    cv::Mat testF = cv::imread("C:\\_Shared\\CZ_FLAG_FULL.png", cv::IMREAD_UNCHANGED);
    cv::cuda::GpuMat inGpu, outGpu;
    inGpu.upload(testF);

    cv::Point2f srcPoints[4];
    cv::Point2f dstPoints[4];
    srcPoints[0] = cv::Point2f{ 0.f, 0.f };
    srcPoints[1] = cv::Point2f{ ((float)testF.cols), 0.f };
    srcPoints[2] = cv::Point2f{ 0.f, (float)testF.rows };
    srcPoints[3] = cv::Point2f{ (float)testF.cols, (float)testF.rows };

    dstPoints[0] = cv::Point2f{ 473.f, 297.f };
    dstPoints[1] = cv::Point2f{ 913.f, 301.f };
    dstPoints[2] = cv::Point2f{ 286.f, 657.f };
    dstPoints[3] = cv::Point2f{ 1109.f, 664.f };

    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);

    cv::Mat testTranformed;
    cv::cuda::warpPerspective(inGpu, outGpu, M, cv::Size{ FRAME_WIDTH, FRAME_HEIGHT });
    outGpu.download(testTranformed);

    cv::namedWindow("f");
    cv:imshow("f", testTranformed);
    cv::waitKey(0);
    return 0;*/

    QApplication a(argc, argv);

    srand(time(nullptr));

    cv::Point2f dstPoints[4];
    

    dstPoints[0] = cv::Point2f{ 473.f, 297.f };
    dstPoints[1] = cv::Point2f{ 913.f, 301.f };
    dstPoints[2] = cv::Point2f{ 286.f, 657.f };
    dstPoints[3] = cv::Point2f{ 1109.f, 664.f };

    QImage im{ "C:\\_Shared\\matting\\frame2.png" };
    im = im.convertToFormat(QImage::Format_RGBA8888);
    uchar* rawData = im.bits();
    auto out = new uchar[im.width() * im.height() * 4];

    uint8_t* graphics = Utils::getImgRawData("C:\\_Shared\\CZ_FLAG_FULL.png", QImage::Format_RGBA8888);
    uint8_t* bgRaw = Utils::getImgRawData("C:\\_Shared\\matting\\bg.png", QImage::Format_RGBA8888);
    //uint8_t* trimap = Utils::getImgRawData("C:\\_Shared\\matting\\trimap2.png", QImage::Format_Grayscale8);

    InsertionGraphicsPipeline pipeline(cv::Size{200, 200}, dstPoints);
    pipeline.initialize(bgRaw);
    pipeline.process(rawData, graphics, out);

    QImage outIm{ out, im.width(), im.height(), QImage::Format_RGB888 };



    //QImage outIm{ outF, im.width(), im.height(), QImage::Format_RGBA8888 };

    QLabel imageContainer;
    imageContainer.setPixmap(QPixmap::fromImage(outIm));
    imageContainer.show();
    //outIm.save("seg3.png");

    //w.show();
    return a.exec();
}
