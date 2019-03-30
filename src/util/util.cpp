#include "util.h"
#include <iostream>

uchar* Utils::getImgRawData(const QString& filePath, QImage::Format format) {
    // TOOD make nicier
    int bytesPerPixel = 1;
    if (format == QImage::Format_RGBA8888)
        bytesPerPixel = 4;
    else if (format == QImage::Format_RGB888)
        bytesPerPixel = 3;

    QImage img{ filePath };
    if (img.isNull()) {
        std::cout << "Loading image " << filePath.toStdString() << " failed." << std::endl;
        return nullptr;
    }
    img = img.convertToFormat(format);

    const size_t bytesCount = img.width() * img.height() * bytesPerPixel;
    uchar* bits = img.bits();
    uchar* copiedData = new uchar[bytesCount];

    std::memcpy(copiedData, bits, bytesCount);

    return copiedData;
};