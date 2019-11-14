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
    Q_PROPERTY(QString homeSlug READ homeSlug WRITE setHomeSlug NOTIFY homeSlugChanged)
    Q_PROPERTY(QString awaySlug READ awaySlug WRITE setAwaySlug NOTIFY awaySlugChanged)

    private:
        QString m_homeSlug;
        QString m_awaySlug;

    public:
        QmlSceneWrapper(QObject* parent = nullptr): QObject{parent} {}

        QString homeSlug() const;
        QString awaySlug() const;

    public:
        void setHomeSlug(const QString& v);
        void setAwaySlug(const QString& v);

    signals:
        void signal();

        void homeSlugChanged(const QString& homeSlug);
        void awaySlugChanged(const QString& awaySlug);
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

        QQuickItem* m_qmlRootItem;
        QQmlContext* m_qmlContext;
        bool m_rendering;
        QmlSceneWrapper* m_sceneWrapper;

        bool initQmlRootItem(QObject* qmlRootObject);

    public:
        QmlRenderer(QObject* parent = nullptr);
        ~QmlRenderer();

        QmlSceneWrapper* sceneWrapper() const;
        void renderFrame();

    private slots:
        void renderNextFrame();

    public slots:
        bool loadQml(const QUrl& url);
        void startGrabbing();
        void startRender();
        void reload();
};