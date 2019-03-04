#include "rtgraphicsinsertion.h"
#include <QCamera>
#include <QCameraInfo>
#include <QVBoxLayout>
#include <qcombobox.h>


RTgraphicsinsertion::RTgraphicsinsertion(QWidget *parent)
	: QWidget(parent)
{
	auto layout = new QVBoxLayout(this);
	m_cameraDevList = new QComboBox;
	layout->setSizeConstraint(QLayout::SetMinimumSize);

	layout->addWidget(m_cameraDevList);

	const auto camerasInfo = QCameraInfo::availableCameras();
	for (auto camInfo : camerasInfo) {
		m_cameraDevList->addItem(camInfo.deviceName());
	}

	this->setLayout(layout);
}
