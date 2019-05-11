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
    m_initializedBgModel = false;
    m_grabbedInitFrameI = 0;
    m_initRequest = false;
    srand(time(nullptr));

    cv::Point2f dstPoints[4];


    dstPoints[0] = cv::Point2f{ 473.f, 297.f };
    dstPoints[1] = cv::Point2f{ 913.f, 301.f };
    dstPoints[2] = cv::Point2f{ 286.f, 657.f };
    dstPoints[3] = cv::Point2f{ 1109.f, 664.f };
    m_pipeline = new InsertionGraphicsPipeline(cv::Size{ 200, 200 }, dstPoints);
    m_graphics = Utils::getImgRawData("C:\\_Shared\\CZ_FLAG_FULL.png", QImage::Format_RGBA8888);

    m_out = new uchar[FRAME_SIZE * Config::CHANNELS_COUNT_INPUT];
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
    const QImage::Format imageFormat = 
        QVideoFrame::imageFormatFromPixelFormat(m_currentFrame.pixelFormat());

    if (imageFormat != QImage::Format_Invalid 
        && m_currentFrame.isValid() && m_currentFrame.map(QAbstractVideoBuffer::ReadOnly)) {
        bool frameFlipped = false;

        if (surfaceFormat().scanLineDirection() == QVideoSurfaceFormat::BottomToTop)
            frameFlipped = true;

        static int i = 1;
        static bool first = true;
        static bool ii = false;

        //std::cout << "Frame: " << (i++) << std::endl;
        // convert video frame to QImage
        QImage im(
            m_currentFrame.bits(),
            m_currentFrame.width(),
            m_currentFrame.height(),
            m_currentFrame.bytesPerLine(),
            imageFormat);

        im = im.convertToFormat(QImage::Format_RGBA8888);
        if (frameFlipped)
            im = im.scaled(QSize(1280, 720)).mirrored();
        if (im.isNull()) {
            m_currentFrame.unmap();
            return;
        }

        if (i < 300) {
            std::cout << "Init Frame: " << (i++) << std::endl;
            m_pipeline->initAddFrame(im.bits());
        }

        else if(!ii) {
            ii = true;
            std::cout << "Frame computing... " << std::endl;
            m_pipeline->computeInitBg();
            std::cout << "Frame computed" << std::endl;
        }

        if(!m_test.isNull())
            painter->drawImage(m_targetRect, m_test, QRect(QPoint(), QSize(1280, 720)));

        // processing
        if (ii) {
            std::cout << "Frame: " << (i++) << std::endl;
            m_pipeline->process(im.bits(), m_graphics, m_out);
            QImage outIm{ m_out, im.width(), im.height(), QImage::Format_RGB888 };
            painter->drawImage(m_targetRect, outIm, QRect(QPoint(), QSize(1280, 720)));
        }
        
        /*else
            painter->drawImage(m_targetRect, im, QRect(QPoint(), QSize(1280, 720)));*/

        /*if (first) {
            first = false;
            m_pipeline->initialize(Utils::getImgRawData(im, QImage::Format_RGBA8888));
        }*/


        m_currentFrame.unmap();
    }
    
}

void VideoProcessingSurface::initBgModel() {
    m_initRequest = true;
}
