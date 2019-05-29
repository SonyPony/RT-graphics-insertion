#pragma once
#include <QObject>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QQuickRenderControl>
#include <QQmlEngine>
#include <QQuickWindow>
#include <QOpenGLFramebufferObject>
#include <QQmlComponent>

class QmlSceneWrapper : public QObject {
    Q_OBJECT

    public:
        QmlSceneWrapper(QObject* parent = nullptr): QObject{parent} {}

    signals:
        void signal();
};

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
        QQmlContext* m_qmlContext;
        bool m_rendering;
        QmlSceneWrapper* m_sceneWrapper;

        bool initQmlRootItem(QObject* qmlRootObject);

    public:
        QmlRenderer(QObject* parent = nullptr);
        ~QmlRenderer();

        QmlSceneWrapper* sceneWrapper() const;
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