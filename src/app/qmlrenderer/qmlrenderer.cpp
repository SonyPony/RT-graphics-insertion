#include "qmlrenderer.h"
#include <QOpenGLFunctions>
#include <QQuickItem>
#include <QTimer>
#include "../../common/config.h"


QmlRenderer::QmlRenderer(QObject* parent) : QObject(parent) {
    m_qmlComponent = nullptr;

    QSurfaceFormat format;
    format.setDepthBufferSize(16);
    format.setStencilBufferSize(8);

    m_context = new QOpenGLContext;
    m_context->setFormat(format);
    m_context->create();

    m_surface = new QOffscreenSurface;
    m_surface->setFormat(m_context->format());
    m_surface->create();

    // QML part init
    m_renderControl = new QQuickRenderControl{ this };
    m_qmlWindow = new QQuickWindow{ m_renderControl };
    m_qmlWindow->setDefaultAlphaBuffer(true);
    m_qmlWindow->setFormat(format);
    m_qmlWindow->setColor(Qt::transparent);

    m_qmlEngine = new QQmlEngine;

    if (!m_qmlEngine->incubationController())
        m_qmlEngine->setIncubationController(m_qmlWindow->incubationController());

    // connect OpenGL with QML
    m_context->makeCurrent(m_surface);
    m_renderControl->initialize(m_context);

    m_fbo = new QOpenGLFramebufferObject{ 
        QSize{GRAPHICS_WIDTH, GRAPHICS_HEIGHT},
        QOpenGLFramebufferObject::CombinedDepthStencil
    };
    m_qmlWindow->setRenderTarget(m_fbo);
}

QmlRenderer::~QmlRenderer() {
    m_context->makeCurrent(m_surface);
    delete m_renderControl;
    delete m_qmlComponent;
    delete m_qmlWindow;
    delete m_qmlEngine;
    delete m_fbo;

    m_context->doneCurrent();

    delete m_surface;
    delete m_context;
}

void QmlRenderer::renderNextFrame() {
    if (!m_context->makeCurrent(m_surface)) {
        qDebug() << "Failed to make current surface.";
        return;
    }

    m_renderControl->polishItems();
    if(m_renderControl->sync())
        m_renderControl->render();

    m_qmlWindow->resetOpenGLState();
    QOpenGLFramebufferObject::bindDefault();

    m_context->functions()->glFlush();

    if (m_fbo->bind()) {
        m_currentFrame = m_fbo->toImage();

        m_fbo->release();
    }

    // request next frame
    QTimer::singleShot(20, this, &QmlRenderer::renderNextFrame);
}

bool QmlRenderer::loadQml(const QUrl& url) {
    m_qmlComponent = new QQmlComponent(m_qmlEngine, url, QQmlComponent::PreferSynchronous);
    if (m_qmlComponent->isError()) {
        for (const auto& error : m_qmlComponent->errors())
            qWarning() << "Error: " << error.url() << " - " << error.line() << ": " << error;
        return false;
    }

    QObject* qmlRootObject = m_qmlComponent->create();
    if (m_qmlComponent->isError()) {
        for (const auto& error : m_qmlComponent->errors())
            qWarning() << "Error: " << error.url() << " - " << error.line() << ": " << error;
        return false;
    }

    QQuickItem* qmlRootItem = qobject_cast<QQuickItem*>(qmlRootObject);
    if (qmlRootItem == nullptr) {
        qWarning() << "Failed conversion to QQuickItem";
        delete qmlRootObject;
        return false;
    }

    qmlRootItem->setParentItem(m_qmlWindow->contentItem());
    qmlRootItem->setSize(QSize{ GRAPHICS_WIDTH, GRAPHICS_HEIGHT });
    m_qmlWindow->setGeometry(0, 0, GRAPHICS_WIDTH, GRAPHICS_HEIGHT);

    return true;
}

void QmlRenderer::start() {
    this->renderNextFrame();
}

QImage QmlRenderer::currentFrame() const {
    return m_currentFrame;
}