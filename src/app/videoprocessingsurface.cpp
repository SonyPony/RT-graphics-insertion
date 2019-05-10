#include "videoprocessingsurface.h"
#include <QVideoSurfaceFormat>
#include <QImage>
#include <QPainter>
#include <QDebug>
#include "../util/util.h"
#include <time.h>

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "opencv2/imgproc.hpp"
#include <chrono>

VideoProcessingSurface::VideoProcessingSurface(QWidget* widget, QObject* parent)
    : QAbstractVideoSurface(parent)
{
    m_widget = widget;

    srand(time(nullptr));

    cv::Point2f dstPoints[4];


    dstPoints[0] = cv::Point2f{ 473.f, 297.f };
    dstPoints[1] = cv::Point2f{ 913.f, 301.f };
    dstPoints[2] = cv::Point2f{ 286.f, 657.f };
    dstPoints[3] = cv::Point2f{ 1109.f, 664.f };
    m_pipeline = new InsertionGraphicsPipeline(cv::Size{ 200, 200 }, dstPoints);
    m_graphics = Utils::getImgRawData("C:\\_Shared\\CZ_FLAG_FULL.png", QImage::Format_RGBA8888);
}


VideoProcessingSurface::~VideoProcessingSurface()
{
}

QList<QVideoFrame::PixelFormat> VideoProcessingSurface::supportedPixelFormats(
    QAbstractVideoBuffer::HandleType handleType) const
{
    if (handleType == QAbstractVideoBuffer::NoHandle) {
        return QList<QVideoFrame::PixelFormat>()
            << QVideoFrame::Format_RGB32
            << QVideoFrame::Format_ARGB32
            << QVideoFrame::Format_ARGB32_Premultiplied
            << QVideoFrame::Format_RGB565
            << QVideoFrame::Format_RGB555;
    }
    else {
        return QList<QVideoFrame::PixelFormat>();
    }
}

bool VideoProcessingSurface::isFormatSupported(const QVideoSurfaceFormat &format) const
{
    QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(format.pixelFormat());
    const QSize size = format.frameSize();


    return imageFormat != QImage::Format_Invalid
        && !size.isEmpty()
        && format.handleType() == QAbstractVideoBuffer::NoHandle;
}

bool VideoProcessingSurface::start(const QVideoSurfaceFormat &format)
{
    const QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(format.pixelFormat());
    const QSize size = format.frameSize();

    if (imageFormat != QImage::Format_Invalid && !size.isEmpty()) {
        m_imageFormat = imageFormat;
        m_imageSize = size;
        m_sourceRect = format.viewport();

        QAbstractVideoSurface::start(format);

        m_widget->updateGeometry();
        updateVideoRect();

        return true;
    }
    else {
        return false;
    }
}

void VideoProcessingSurface::stop()
{
    m_currentFrame = QVideoFrame();
    m_targetRect = QRect();

    QAbstractVideoSurface::stop();

    m_widget->update();
}

void VideoProcessingSurface::updateVideoRect()
{
    QSize size = QSize(1280, 720);//surfaceFormat().sizeHint();
    size.scale(m_widget->size().boundedTo(size), Qt::KeepAspectRatio);

    m_targetRect = QRect(QPoint(0, 0), size);
    m_targetRect.moveCenter(m_widget->rect().center());
}

bool VideoProcessingSurface::present(const QVideoFrame &frame)
{
    if (surfaceFormat().pixelFormat() != frame.pixelFormat()
        || surfaceFormat().frameSize() != frame.size()) {
        setError(IncorrectFormatError);
        stop();

        return false;
    }
    else {
        m_currentFrame = frame;

        m_widget->repaint(m_targetRect);

        return true;
    }
}

void VideoProcessingSurface::paint(QPainter *painter)
{
    if (m_currentFrame.map(QAbstractVideoBuffer::ReadOnly)) {
        const QTransform oldTransform = painter->transform();

        if (surfaceFormat().scanLineDirection() == QVideoSurfaceFormat::BottomToTop) {
            painter->scale(1, -1);
            painter->translate(0, -m_widget->height());
        }

        QImage image(
            m_currentFrame.bits(),
            m_currentFrame.width(),
            m_currentFrame.height(),
            m_currentFrame.bytesPerLine(),
            m_imageFormat);
        image = image.scaled(QSize(1280, 720));


        // processing
   
        QImage im = image;//{ "C:\\_Shared\\matting\\frame2.png" };
        im = im.convertToFormat(QImage::Format_RGBA8888);
        uchar* rawData = im.bits();
        auto out = new uchar[im.width() * im.height() * 4];


        
        //uint8_t* bgRaw = Utils::getImgRawData("C:\\_Shared\\matting\\bg.png", QImage::Format_RGBA8888);
        
        static bool first = true;
        if (first) {
            first = false;
            uint8_t* bgRaw = Utils::getImgRawData(im, QImage::Format_RGBA8888);
            m_pipeline->initialize(bgRaw);
        }
        
        
        m_pipeline->process(rawData, m_graphics, out);

        //Format_RGB888
        QImage outIm{ out, im.width(), im.height(), QImage::Format_RGB888 };
        //--------------------

        painter->drawImage(m_targetRect, outIm, QRect(QPoint(), QSize(1280, 720)));

        painter->setTransform(oldTransform);

        m_currentFrame.unmap();
    }
}