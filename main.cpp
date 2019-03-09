#include "rtgraphicsinsertion.h"
#include <QtWidgets/QApplication>
#include "vibe.cuh"
#include <iostream>
#include <QLabel>

using namespace std;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	//RTgraphicsinsertion w;

	QImage im{ "C:\\_Shared\\000002.jpg" };
	QImage bg{ "C:\\_Shared\\bg-litovel.png" };
	im = im.convertToFormat(QImage::Format_RGBA8888);
	uchar* rawData = im.bits();
	auto out = new uchar[im.width() * im.height()];

	memset(out, 127, im.width() * im.height());

	bg = bg.convertToFormat(QImage::Format_RGBA8888);
	uint8_t* bgRaw = bg.bits();

	GPU::InsertionGraphicsPipeline pipeline{im.width(), im.height()};
	//pipeline.process(rawData, nullptr, out);
	pipeline.initialize(bgRaw);
	pipeline.process(rawData, nullptr, out);

	cudaDeviceSynchronize();
	QImage outIm{ out, im.width(), im.height(), QImage::Format_Grayscale8 };

	QLabel imageContainer;
	imageContainer.setPixmap(QPixmap::fromImage(outIm));
	imageContainer.show();

	//w.show();
	return a.exec();
}
