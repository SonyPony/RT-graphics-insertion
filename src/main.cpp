#include "pipeline/insertiongraphicspipeline.cuh"
#include <QtWidgets/QApplication>
#include "pipeline/segmentation/vibe.cuh"
#include <iostream>
#include <QLabel>
#include <QDebug>
#include "util/util.h"
#include <time.h>

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "opencv2/imgproc.hpp"
#include <chrono>
#include "app/rtwindow.h"

using namespace std;

using namespace cv;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    RTWindow appWindow;

    srand(time(nullptr));

    appWindow.show();
    return a.exec();
}
