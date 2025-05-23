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
    m_computedTransM = false;
    m_initFrameIndex = 0;
    m_initRequest = false;
    m_inited = false;
    m_exportBgRequest = false;
    srand(time(nullptr));

    m_pipeline = new InsertionGraphicsPipeline;
    m_out = new uchar[FRAME_SIZE * Config::CHANNELS_COUNT_INPUT];
    m_bgModelBuffer = new uchar[FRAME_SIZE * Config::CHANNELS_COUNT_INPUT];

    m_pipeline->createGraphicsResource(1, cudaGraphicsMapFlagsReadOnly);
}

VideoProcessingSurface::~VideoProcessingSurface()
{
}

void VideoProcessingSurface::setTransformPoints(cv::Size graphicsSize, cv::Point2f dstPoints[]) {
    m_computedTransM = true;
    m_pipeline->computeTransMatrix(graphicsSize, dstPoints);
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
        updateVideoRect(QRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT));

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

void VideoProcessingSurface::updateVideoRect(const QRect& rect)
{
    m_targetRect = rect;
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

        QImage im(
            m_currentFrame.bits(),
            m_currentFrame.width(),
            m_currentFrame.height(),
            m_currentFrame.bytesPerLine(),
            imageFormat);

        im = im.convertToFormat(QImage::Format_RGBA8888);
        if (frameFlipped)
            ;// im = im.scaled(QSize(FRAME_WIDTH, FRAME_HEIGHT)).mirrored();
        if (im.isNull()) {
            m_currentFrame.unmap();
            return;
        }

        if (m_initRequest && m_initFrameIndex < VideoProcessingSurface::INIT_FRAMES_COUNT) {
            std::cout << "Init Frame: " << (m_initFrameIndex++) << std::endl;
            m_pipeline->initAddFrame(im.bits());
        }

        else if(m_initRequest && m_initFrameIndex >= VideoProcessingSurface::INIT_FRAMES_COUNT
        ) {
            m_inited = true;
            m_initRequest = false;

            std::cout << "Frame computing... " << std::endl;
            m_pipeline->computeInitBg(m_out);
            std::cout << "Frame computed" << std::endl;
            QImage bgModel{ m_out, im.width(), im.height(), QImage::Format_RGBA8888 };
            bgModel.save("bg-models/bg.png");
        }

        if (m_inited && !m_computedTransM) {
            QImage bgModel{ m_out, im.width(), im.height(), QImage::Format_RGBA8888 };
            painter->drawImage(
                m_targetRect,
                bgModel,
                QRect(QPoint(), QSize(FRAME_WIDTH, FRAME_HEIGHT))
            );
        }

        // processing
        if (!m_initRequest && m_inited && m_computedTransM) {
            using namespace std::chrono;
            m_pipeline->process(
                im.bits(), 
                m_out, 
                (m_exportBgRequest) ?m_bgModelBuffer :nullptr
            );
            
            QImage outIm{ m_out, im.width(), im.height(), QImage::Format_RGB888 };
            painter->drawImage(
                m_targetRect, 
                outIm, 
                QRect(QPoint(), QSize(FRAME_WIDTH, FRAME_HEIGHT))
            );

            if (m_exportBgRequest) {
                QImage bgModel{ m_bgModelBuffer, FRAME_WIDTH, FRAME_HEIGHT, QImage::Format_RGBA8888 };
                bgModel.save("bg-models/bg_temp.png");

                std::cout << "Model saved." << std::endl;
                m_exportBgRequest = false;
            }
        }

        m_currentFrame.unmap();
    }
    
}

void VideoProcessingSurface::initBgModelFromImage(const QImage & img)
{
    m_inited = true;
    m_initRequest = false;

    QImage bg = img.convertToFormat(QImage::Format_RGBA8888);
    memcpy(m_out, bg.bits(), 4 * FRAME_SIZE);
    std::cout << "Background model loaded. " << std::endl;
    m_pipeline->initBgFromImage(bg.bits());
}

void VideoProcessingSurface::initBgModel() {
    m_initFrameIndex = 0;
    m_initRequest = true;
    m_inited = false;
    m_computedTransM = false;
}

void VideoProcessingSurface::exportBgModel() {
    m_exportBgRequest = true;
}