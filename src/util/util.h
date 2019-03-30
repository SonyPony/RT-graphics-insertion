#ifndef UTIL
#define UTIL

#include <QString>
#include <QImage>
#include <string.h>

namespace Utils {
    uchar* getImgRawData(const QString& filePath, QImage::Format format);
}

#endif // !UTIL