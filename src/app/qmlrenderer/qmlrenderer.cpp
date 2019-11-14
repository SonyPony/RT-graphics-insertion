#include "qmlrenderer.h"
#include <QOpenGLFunctions>
#include <QQuickItem>
#include <QTimer>
#include "../../common/config.h"
#include <qqmlcontext.h>


QmlRenderer::QmlRenderer(QObject* parent) : QObject(parent) {
    m_qmlComponent = nullptr;
    m_qmlRootItem = nullptr;
    m_rendering = false;

    m_sceneWrapper = new QmlSceneWrapper{ this };

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

    m_qmlContext = new QQmlContext{ m_qmlEngine->rootContext() };
    m_qmlContext->setContextProperty("sceneWrapper", m_sceneWrapper);

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


void QmlRenderer::renderFrame()
{
    if (!m_rendering)
        return;
    if (!m_context->makeCurrent(m_surface)) {
        qDebug() << "Failed to make current surface.";
        return;
    }

    m_renderControl->polishItems();

    if (m_renderControl->sync())
        m_renderControl->render();
    
    m_qmlWindow->resetOpenGLState();
    QOpenGLFramebufferObject::bindDefault();

    m_context->functions()->glFlush();
}

void QmlRenderer::renderNextFrame() {
    this->renderFrame();
    // request next frame
    QTimer::singleShot(40, this, &QmlRenderer::renderNextFrame);
}

bool QmlRenderer::loadQml(const QUrl& url) {
    m_qmlComponent = new QQmlComponent(m_qmlEngine, url, QQmlComponent::PreferSynchronous);
    if (m_qmlComponent->isError()) {
        for (const auto& error : m_qmlComponent->errors())
            qWarning() << "Error: " << error.url() << " - " << error.line() << ": " << error;
        return false;
    }

    QObject* qmlRootObject = m_qmlComponent->create(m_qmlContext);
    if(!this->initQmlRootItem(qmlRootObject))
        return false;

    return true;
}

void QmlRenderer::startRender() {
    m_rendering = true;
}

void QmlRenderer::startGrabbing() {
    this->startRender();
    this->renderNextFrame();
}

void QmlRenderer::reload() {
    const QUrl qmlUrl = m_qmlComponent->url();
    m_qmlRootItem->deleteLater();

    qDebug() << "Reloading..." << qmlUrl;
    m_qmlEngine->clearComponentCache();
    m_qmlComponent->loadUrl(qmlUrl, QQmlComponent::PreferSynchronous);
    this->initQmlRootItem(m_qmlComponent->create(m_qmlContext));
}

QmlSceneWrapper * QmlRenderer::sceneWrapper() const
{
    return m_sceneWrapper;
}

bool QmlRenderer::initQmlRootItem(QObject* qmlRootObject)
{
    if (m_qmlComponent->isError()) {
        for (const auto& error : m_qmlComponent->errors())
            qWarning() << "Error: " << error.url() << " - " << error.line() << ": " << error;
        return false;
    }

    m_qmlRootItem = qobject_cast<QQuickItem*>(qmlRootObject);

    if (m_qmlRootItem == nullptr) {
        qWarning() << "Failed conversion to QQuickItem";
        delete qmlRootObject;
        return false;
    }

    m_qmlRootItem->setParentItem(m_qmlWindow->contentItem());
    m_qmlRootItem->setSize(QSize{ GRAPHICS_WIDTH, GRAPHICS_HEIGHT });
    m_qmlWindow->setGeometry(0, 0, GRAPHICS_WIDTH, GRAPHICS_HEIGHT);

    return true;
}

QString QmlSceneWrapper::homeSlug() const
{
    return m_homeSlug;
}

QString QmlSceneWrapper::awaySlug() const
{
    return m_awaySlug;
}

void QmlSceneWrapper::setHomeSlug(const QString & v)
{
    if (m_homeSlug == v)
        return;

    m_homeSlug = v;
    emit this->homeSlugChanged(v);
}

void QmlSceneWrapper::setAwaySlug(const QString & v)
{
    if (m_awaySlug == v)
        return;

    m_awaySlug = v;
    emit this->awaySlugChanged(v);
}
