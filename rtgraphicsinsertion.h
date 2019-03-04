#pragma once

#include <QWidget>
#include <QComboBox>


class RTgraphicsinsertion : public QWidget
{
	Q_OBJECT

public:
	RTgraphicsinsertion(QWidget *parent = Q_NULLPTR);

private:
	QComboBox* m_cameraDevList;
};
