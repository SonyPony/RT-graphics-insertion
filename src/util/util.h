#ifndef UTIL
#define UTIL

#include <QString>
#include <QImage>
#include <string.h>

namespace Utils {
    uchar* getImgRawData(const QString& filePath, QImage::Format format);
    uchar* getImgRawData(const QImage& img, QImage::Format format);
}

#endif // !UTIL