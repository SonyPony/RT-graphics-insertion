#include "rtwindow.h"
#include <QDebug>
#include <QHBoxLayout>
#include <QCameraInfo>
#include <QPainter>
#include <QPaintEvent>
#include <QVideoSurfaceFormat>


RTWindow::RTWindow(QWidget* parent): QWidget(parent)
{
    m_currentCamera = nullptr;
    m_processing = new VideoProcessingSurface(this);

    auto layout = new QHBoxLayout(this);
    m_cameraSelection = new QComboBox{ this };
    
    layout->addWidget(m_cameraSelection);
    this->setLayout(layout);
    
    // add camera selections items to combobox
    for (const auto& cameraInfo : QCameraInfo::availableCameras()) {
        m_cameraSelection->addItem(cameraInfo.deviceName());
        m_camerasList.append(new QCamera(cameraInfo, this));
    }

    // connections
    connect(m_cameraSelection, QOverload<int>::of(&QComboBox::activated),
        [this](int index) {
        m_currentCamera = m_camerasList[index];
        qDebug() << m_currentCamera;
        m_currentCamera->setViewfinder(m_processing);
        m_currentCamera->start();
    });

    

}

RTWindow::~RTWindow()
{
}

QSize RTWindow::sizeHint() const
{
    return m_processing->surfaceFormat().sizeHint();
}
//! [2]


//! [3]
void RTWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    if (m_processing->isActive()) {
        const QRect videoRect = m_processing->videoRect();

        if (!videoRect.contains(event->rect())) {
            QRegion region = event->region();
            region = region.subtracted(videoRect);

            QBrush brush = palette().background();

            for (const QRect &rect : region)
                painter.fillRect(rect, brush);
        }

        m_processing->paint(&painter);
    }
    else {
        painter.fillRect(event->rect(), palette().background());
    }
}

void RTWindow::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);

    m_processing->updateVideoRect();
}
