#pragma once
#include <QWidget>
#include <QComboBox> 
#include <QCameraInfo>
#include "videoprocessingsurface.h"
#include <QQuickWidget>
#include "qmlrenderer/qmlrenderer.h"


class RTWindow : public QWidget
{
    Q_OBJECT

    private:
        QList<QCamera*> m_camerasList;
        QComboBox* m_cameraSelection;
        VideoProcessingSurface* m_processing;
        QQuickWidget* m_transformView;
        QmlRenderer* m_graphicsRenderer;

        QCamera* m_currentCamera;

    public:
        RTWindow(QWidget* parent = nullptr);
        ~RTWindow();
        QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
};

