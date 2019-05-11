#pragma once
#include <QAbstractVideoSurface>
#include <QWidget>
#include "../pipeline/insertiongraphicspipeline.cuh"


class VideoProcessingSurface: public QAbstractVideoSurface
{
    Q_OBJECT
    
    private:
        QWidget* m_widget;
        QImage::Format m_imageFormat;
        QRect m_targetRect;
        QSize m_imageSize;
        QRect m_sourceRect;
        QVideoFrame m_currentFrame;
        InsertionGraphicsPipeline* m_pipeline;
        uint8_t* m_graphics;

        uint8_t* m_out;

    public:
        VideoProcessingSurface(QWidget* widget, QObject* parent = nullptr);
        ~VideoProcessingSurface();

        QList<QVideoFrame::PixelFormat> supportedPixelFormats(
            QAbstractVideoBuffer::HandleType handleType = QAbstractVideoBuffer::NoHandle) const override;
        bool isFormatSupported(const QVideoSurfaceFormat &format) const override;

        bool start(const QVideoSurfaceFormat &format) override;
        void stop() override;

        bool present(const QVideoFrame &frame) override;
        QRect videoRect() const { return m_targetRect; }
        void updateVideoRect();

        void paint(QPainter *painter);
};

