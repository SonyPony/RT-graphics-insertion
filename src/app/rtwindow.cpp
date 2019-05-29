#include "rtwindow.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QCameraInfo>
#include <QPainter>
#include <QPaintEvent>
#include <QVideoSurfaceFormat>
#include <QQuickWidget>
#include <QPushButton>
#include <QQmlEngine>
#include "qmlrenderer/qmlrenderer.h"
#include <QQuickItem>
#include <qlist.h>


RTWindow::RTWindow(QWidget* parent): QWidget(parent)
{
    m_graphicsRenderer = new QmlRenderer(this);
    m_graphicsRenderer->loadQml(QUrl::fromLocalFile(
        "C:/Users/Sony/source/repos/RT-graphics-insertion/RT-graphics-insertion/src/scene/main.qml"
    ));

    m_currentCamera = nullptr;
    m_processing = new VideoProcessingSurface(this);

    m_transformView = new QQuickWidget(this);
    m_transformView->setVisible(false);
    m_transformView->setClearColor(Qt::transparent);
    m_transformView->setAttribute(Qt::WA_AlwaysStackOnTop, true);
    m_transformView->setResizeMode(QQuickWidget::SizeViewToRootObject);
    m_transformView->setSource(QUrl::fromLocalFile(":/RTgraphicsinsertion/qml/main.qml"));
    
    auto layout = new QVBoxLayout(this);
    m_cameraSelection = new QComboBox{ this };

    m_confirmButton = new QPushButton("Confim", this);
    m_confirmButton->setVisible(false);

    m_transformButton = new QPushButton("Transform", this);
    m_transformButton->setVisible(false);
    
    m_initBgButton = new QPushButton("Init", this);
    m_initBgButton->setVisible(false);

    m_reloadSceneButton = new QPushButton("Reload scene", this);
    m_reloadSceneButton->setVisible(false);

    for (auto w : QList<QWidget*>{ 
        m_initBgButton, m_transformButton,
        m_confirmButton, m_reloadSceneButton 
    }) {
        w->setMaximumWidth(150);
    }

    layout->setSpacing(0);
    layout->setMargin(0);
    layout->addWidget(m_cameraSelection);
    layout->addWidget(m_transformView);
    layout->addWidget(m_initBgButton, 0, Qt::AlignHCenter);
    layout->addWidget(m_transformButton, 0, Qt::AlignHCenter);
    layout->addWidget(m_reloadSceneButton, 0, Qt::AlignHCenter);
    layout->addWidget(m_confirmButton, 0, Qt::AlignHCenter);
    this->setLayout(layout);

    // add camera selections items to combobox
    for (const auto& cameraInfo : QCameraInfo::availableCameras()) {
        m_cameraSelection->addItem(cameraInfo.deviceName());
        m_camerasList.append(new QCamera(cameraInfo, this));
    }

    connect(m_reloadSceneButton, &QPushButton::clicked, [this]() {
        m_graphicsRenderer->reload();
    });

    connect(m_initBgButton, &QPushButton::clicked, [this]() {
        m_processing->initBgModel();
    });

    connect(m_confirmButton, &QPushButton::clicked, [this]() {
        m_confirmButton->setVisible(false);
        m_transformButton->setVisible(false);
        m_transformView->setVisible(false);
        m_initBgButton->setVisible(false);
        m_reloadSceneButton->setVisible(false);
        m_processing->updateVideoRect(QRect(0, 0, 1920, 1080));
        this->setWindowState(Qt::WindowFullScreen);
    });

    connect(m_transformButton, &QPushButton::clicked, [this]() {
        QQuickItem* rootItem = m_transformView->rootObject();
        if (rootItem != nullptr) {
            QVariantList varPoints = rootItem->property("transformPoints").value<QJSValue>().toVariant().toList();
            cv::Point2f dstPoints[4];

            for (int i = 0; i < 4; i++) {
                const QPointF qpoint = varPoints.at(i).toPointF();
                dstPoints[i] = cv::Point2f{ 
                    (float)qpoint.x(),
                    (float)qpoint.y()
                };
            }

            m_processing->setTransformPoints(cv::Size{ GRAPHICS_WIDTH, GRAPHICS_HEIGHT }, dstPoints);
        }
    });

    // connections
    connect(m_cameraSelection, QOverload<int>::of(&QComboBox::activated),
        [this](int index) {
        m_currentCamera = m_camerasList[index];
        m_currentCamera->load();
        qDebug() << m_currentCamera->supportedViewfinderResolutions();
        QCameraViewfinderSettings viewFinderSettings;
        viewFinderSettings.setMinimumFrameRate(15);
        viewFinderSettings.setMaximumFrameRate(25);
        viewFinderSettings.setResolution(1920, 720);

        m_currentCamera->setViewfinder(m_processing);
        m_currentCamera->start();
        m_cameraSelection->setVisible(false);

        m_transformView->setVisible(true);
        m_confirmButton->setVisible(true);
        m_transformButton->setVisible(true);
        m_initBgButton->setVisible(true);
        m_reloadSceneButton->setVisible(true);

        m_graphicsRenderer->start();
    });

}

RTWindow::~RTWindow()
{
}

QSize RTWindow::sizeHint() const
{
    return m_processing->surfaceFormat().sizeHint();
}

void RTWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    //auto start = std::chrono::high_resolution_clock::now();
    if (m_processing->isActive()) {
        const QRect videoRect = m_processing->videoRect();

        if (!videoRect.contains(event->rect())) {
            QRegion region = event->region();
            region = region.subtracted(videoRect);

            QBrush brush = palette().background();

            for (const QRect &rect : region)
                painter.fillRect(rect, brush);
        }

        m_processing->paint(&painter, m_graphicsRenderer->currentFrame());
    }
    else {
        painter.fillRect(event->rect(), palette().background());
    }
    //using namespace std::chrono;
    //qDebug() <<  "Whole: " << (duration_cast<milliseconds>(high_resolution_clock::now() - start).count());
}

