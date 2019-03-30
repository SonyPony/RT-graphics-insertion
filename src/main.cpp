#include "pipeline/insertiongraphicspipeline.cuh"
#include <QtWidgets/QApplication>
#include "pipeline/segmentation/vibe.cuh"
#include <iostream>
#include <QLabel>
#include <QDebug>
#include "util/util.h"

using namespace std;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //RTgraphicsinsertion w;

    QImage im{ "C:\\_Shared\\matting\\frame2.png" };
    im = im.convertToFormat(QImage::Format_RGBA8888);
    uchar* rawData = im.bits();
    auto out = new uchar[im.width() * im.height()];

    uint8_t* bgRaw = Utils::getImgRawData("C:\\_Shared\\matting\\bg.png", QImage::Format_RGBA8888);
    uint8_t* trimap = Utils::getImgRawData("C:\\_Shared\\matting\\trimap2.png", QImage::Format_Grayscale8);

    //GPU::InsertionGraphicsPipeline pipeline{im.width(), im.height()};
    Gpu::InsertionGraphicsPipeline pipeline(im.width(), im.height());
    pipeline.initialize(bgRaw);
    pipeline.process(rawData, nullptr, out);

    /*pipeline.initialize(bgRaw);
    pipeline.process(rawData, nullptr, out, trimap);*/

    cudaDeviceSynchronize();
    QImage outIm{ out, im.width(), im.height(), QImage::Format_Grayscale8 };

    QLabel imageContainer;
    imageContainer.setPixmap(QPixmap::fromImage(outIm));
    imageContainer.show();
    outIm.save("seg2.png");

    //w.show();
    return a.exec();
}
