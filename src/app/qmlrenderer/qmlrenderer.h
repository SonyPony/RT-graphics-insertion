#pragma once
#include <QObject>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QQuickRenderControl>
#include <QQmlEngine>
#include <QQuickWindow>
#include <QOpenGLFramebufferObject>
#include <QQmlComponent>


class QmlRenderer : public QObject
{
    Q_OBJECT

    private:
        QOpenGLFramebufferObject* m_fbo;
        QOpenGLContext* m_context;
        QOffscreenSurface* m_surface;
        QQuickRenderControl* m_renderControl;

        QQmlEngine* m_qmlEngine;
        QQmlComponent* m_qmlComponent;
        QQuickWindow* m_qmlWindow;
        QImage m_currentFrame;

        QQuickItem* m_qmlRootItem;
        bool m_rendering;

        bool initQmlRootItem(QObject* qmlRootObject);

    public:
        QmlRenderer(QObject* parent = nullptr);
        ~QmlRenderer();

        QImage currentFrame() const;
        QImage renderFrame();

    private slots:
        void renderNextFrame();

    public slots:
        bool loadQml(const QUrl& url);
        void startGrabbing();
        void startRender();
        void reload();
};