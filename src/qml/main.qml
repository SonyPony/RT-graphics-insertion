import QtQuick 2.7
import "view" as View

Item {
    property alias transformPoints: transformView.points

    width: 1920
    height: 1080

    /*Rectangle {
        opacity: 0.5
        color: "white"
        anchors.fill: parent
    }*/

    Shortcut {
        sequence: "Ctrl+C"
        onActivated: transformView.reset()
    }

    View.TransformView {
        id: transformView
        initPoints: [Qt.point(0, 0), Qt.point(400, 0), Qt.point(400, 400), Qt.point(0, 400)]
        color: "blue"
        anchors.fill: parent
    }
}
