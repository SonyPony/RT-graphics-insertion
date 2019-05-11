import QtQuick 2.0

Item {
    id: component

    signal reset()

    property int cornerSize: 18
    property color color: "black"
    property int lineWidth: 3
    property alias initPoints: repeater.model
    property var points: [Qt.point(0, 0), Qt.point(20, 0), Qt.point(20, 20), Qt.point(0, 20)]

    QtObject {
        id: internal
        function updatePoints(index, x, y) {
            component.points[index] = Qt.point(x, y)
            component.pointsChanged()
        }
    }

    Canvas {
        id: canvas
        opacity: 0.3
        anchors.fill: parent

        onPaint: {
            var item = repeater.itemAt(0)
            var itemOffset = Qt.point(item.width / 2, item.height / 2)
            var ctx = canvas.getContext("2d")
            ctx.clearRect(0, 0, width, height)

            ctx.beginPath()
            ctx.fillStyle = component.color
            ctx.lineJoin = "bevel"
            ctx.lineWidth = component.lineWidth
            ctx.moveTo(item.x + itemOffset.x, item.y + itemOffset.y)
            for(var i = 0; i < repeater.count; i++) {
                item = repeater.itemAt(i)
                ctx.lineTo(item.x + itemOffset.x, item.y + itemOffset.y)
            }
            item = repeater.itemAt(0)
            ctx.lineTo(item.x + itemOffset.x, item.y + itemOffset.y)

            ctx.closePath()
            ctx.fill()
        }
    }

    Repeater {
        id: repeater

        Rectangle {
            id: corner

            readonly property int centerOffset: width / 2

            x: modelData.x - width / 2
            y: modelData.y - height / 2
            width: component.cornerSize
            height: width
            radius: width
            color: component.color
            opacity: 0.5

            Drag.active: dragArea.drag.active
            Drag.hotSpot: Qt.point(width / 2, height / 2)

            onXChanged: {
                internal.updatePoints(index, x + centerOffset, y + centerOffset)
                canvas.requestPaint()
            }

            onYChanged: {
                internal.updatePoints(index, x + centerOffset, y + centerOffset)
                canvas.requestPaint()
            }

            MouseArea {
                id: dragArea

                anchors.fill: parent
                drag.target: parent
            }

            function forceRepos(i) {
                corner.x = component.initPoints[i].x
                corner.y = component.initPoints[i].y
            }

            Connections {
                target: component
                onReset: forceRepos(index)
            }
        }
    }
}
