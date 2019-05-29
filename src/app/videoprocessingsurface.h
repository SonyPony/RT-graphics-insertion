#pragma once
#include <QAbstractVideoSurface>
#include <QWidget>
#include "../pipeline/insertiongraphicspipeline.cuh"


class VideoProcessingSurface : public QAbstractVideoSurface
{
    Q_OBJECT

    public:
        static constexpr int INIT_FRAMES_COUNT = 300;

    private:
        QWidget* m_widget;
        QImage::Format m_imageFormat;
        QRect m_targetRect;
        QSize m_imageSize;
        QRect m_sourceRect;
        QVideoFrame m_currentFrame;
        InsertionGraphicsPipeline* m_pipeline;

        uint8_t* m_out;
        bool m_computedTransM;
        bool m_initRequest;
        int m_initFrameIndex;
        bool m_inited;

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
        void updateVideoRect(const QRect& rect);

        void setTransformPoints(cv::Size graphicsSize, cv::Point2f dstPoints[]);

        void paint(QPainter *painter, const QImage& graphics);

    public slots:
        void initBgModel();
        void initBgModelFromImage(const QImage& img);
};

