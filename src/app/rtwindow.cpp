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
#include <QShortcut>
#include <QFileDialog>
#include <QFormLayout>
#include <QScreen>
#include <QApplication>


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

    m_confirmButton = new QComboBox(this);
    m_transformButton = new QPushButton("Transform", this);
    m_initBgButton = new QPushButton("Init", this);
    m_loadBgModel = new QPushButton("Load Bg", this);

    layout->setSpacing(0);
    layout->setMargin(0);
    layout->addWidget(m_cameraSelection);
    layout->addWidget(m_transformView);

    for (auto w : QList<QWidget*>{ 
        m_initBgButton, m_transformButton,
        m_loadBgModel, m_confirmButton
    }) {
        w->setMaximumWidth(150);
        w->setVisible(false);
        layout->addWidget(w, 0, Qt::AlignHCenter);
    }

    this->setLayout(layout);

    // qml scene controls
    const QStringList slugList{
        "dukla-praha", "hk-ivancice", "horka-nad-moravou", "kostelec-na-hane",
        "lions-hostivice", "litovel-minicup", "praha-chodov", "sk-zeravice",
        "sokol-sokolnice", "sokol-telnice", "tatran-litovel",
        "tj-nachod", "valasske-mezirici", "velka-bystrice", "velke-mezirici"
    };
    auto qmlControls = new QWidget;
    auto qmlControlsLayout = new QVBoxLayout{ qmlControls };

    auto qmlSendSignalButton = new QPushButton{ "Send qml signal" };
    qmlSendSignalButton->setMinimumSize(300, 300);
    qmlControlsLayout->addWidget(qmlSendSignalButton);

    auto qmlSlugsLayout = new QFormLayout;
    auto qmlHomeSlugSelection = new QComboBox;
    auto qmlAwaySlugSelection = new QComboBox;

    qmlHomeSlugSelection->addItems(slugList);
    qmlAwaySlugSelection->addItems(slugList);
    m_graphicsRenderer->sceneWrapper()->setHomeSlug(qmlHomeSlugSelection->currentText());
    m_graphicsRenderer->sceneWrapper()->setAwaySlug(qmlAwaySlugSelection->currentText());

    qmlSlugsLayout->addRow("Home:", qmlHomeSlugSelection);
    qmlSlugsLayout->addRow("Away:", qmlAwaySlugSelection);

    qmlControlsLayout->addLayout(qmlSlugsLayout);
    
    qmlControls->show();

    QStringList screensList;
    for (auto s : QApplication::screens())
        screensList.append(s->name());
    m_confirmButton->addItems(screensList);

    connect(qmlHomeSlugSelection, qOverload<const QString&>(&QComboBox::activated),
        m_graphicsRenderer->sceneWrapper(), &QmlSceneWrapper::setHomeSlug);
    connect(qmlAwaySlugSelection, qOverload<const QString&>(&QComboBox::activated),
        m_graphicsRenderer->sceneWrapper(), &QmlSceneWrapper::setAwaySlug);
    connect(qmlSendSignalButton, &QPushButton::clicked, 
        m_graphicsRenderer->sceneWrapper(), &QmlSceneWrapper::signal);

    // add camera selections items to combobox
    for (const auto& cameraInfo : QCameraInfo::availableCameras()) {
        m_cameraSelection->addItem(cameraInfo.deviceName());
        m_camerasList.append(new QCamera(cameraInfo, this));
    }

    auto reloadQmlShortcut = new QShortcut{ QKeySequence{"Ctrl+R"}, this };
    connect(reloadQmlShortcut, &QShortcut::activated, [this]() {
        m_graphicsRenderer->reload();
    });

    auto saveBgShortcut = new QShortcut{ QKeySequence{"Ctrl+S"}, this };
    connect(saveBgShortcut, &QShortcut::activated, [this]() {
        m_processing->exportBgModel();
    });

    connect(m_loadBgModel, &QPushButton::clicked, [this]() {
        const QString filename = QFileDialog::getOpenFileName(
            this, "Load background model", QString{}, "Image Files (*.png)"
        );
        QImage bgModel{ filename };
        if(!bgModel.isNull())
            m_processing->initBgModelFromImage(bgModel);
    });

    connect(m_initBgButton, &QPushButton::clicked, [this]() {
        m_processing->initBgModel();
    });

    connect(m_confirmButton, qOverload<const QString&>(&QComboBox::activated), [this](const QString& screenName) {
        for (int i = 0; i < QApplication::screens().count(); i++) {
            QScreen* screen = QApplication::screens().at(i);
            if (screen->name() == screenName) {
                qDebug() << "Show on screen" << screenName << screen->geometry();
                this->windowHandle()->setScreen(screen);
            }
        }
        
        m_confirmButton->setVisible(false);
        m_transformButton->setVisible(false);
        m_transformView->setVisible(false);
        m_initBgButton->setVisible(false);
        m_loadBgModel->setVisible(false);
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
        m_loadBgModel->setVisible(true);

        m_graphicsRenderer->startRender();
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

        m_graphicsRenderer->renderFrame();
        m_processing->paint(&painter);
    }
    else {
        painter.fillRect(event->rect(), palette().background());
    }
    //using namespace std::chrono;
    //qDebug() <<  "Whole: " << (duration_cast<milliseconds>(high_resolution_clock::now() - start).count());
}

