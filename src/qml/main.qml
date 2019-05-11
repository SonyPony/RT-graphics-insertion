import QtQuick 2.7
import "view" as View

Item {
    width: 500
    height: 500

    /*Rectangle {
        opacity: 0.5
        color: "white"
        anchors.fill: parent
    }*/

    Shortcut {
        sequence: "Ctrl+R"
        onActivated: transformView.reset()
    }

    View.TransformView {
        id: transformView
        initPoints: [Qt.point(0, 0), Qt.point(20, 0), Qt.point(20, 20), Qt.point(0, 20)]
        color: "orange"
        anchors.fill: parent

        //onPointsChanged: Adapter.transformPoints = points
    }
}
