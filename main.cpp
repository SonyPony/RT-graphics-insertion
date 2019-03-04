#include "rtgraphicsinsertion.h"
#include <QtWidgets/QApplication>
#include "vibe.cuh"
#include <iostream>
#include <QLabel>

using namespace std;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	RTgraphicsinsertion w;

	QImage im{ "C:\\_Shared\\000002.jpg" };
	im = im.convertToFormat(QImage::Format_ARGB32);
	uchar* rawData = im.bits();
	auto out = new uchar[4 * im.width() * im.height()];

	memset(out, 255, 4 * im.width() * im.height());

	GPU::InsertionGraphicsPipeline pipeline{im.width(), im.height()};
	pipeline.process(rawData, nullptr, out);

	cudaDeviceSynchronize();
	QImage outIm{ out, im.width(), im.height(), QImage::Format_ARGB32 };

	QLabel imageContainer;
	imageContainer.setPixmap(QPixmap::fromImage(outIm));
	imageContainer.show();

	w.show();
	return a.exec();
}
