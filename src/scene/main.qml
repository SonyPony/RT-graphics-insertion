import QtQuick 2.7

Item {
    width: 200
    height: 200

    Rectangle {
        id: r
        color: "orange"
        width: 20;
        height: 20

        Component.onCompleted: SequentialAnimation {
            loops: Animation.Infinite
            running: true
            NumberAnimation { target: r; property: "x"; to: 180; duration: 400 }
            NumberAnimation { target: r; property: "x"; to: 0; duration: 400 }
        }
    }

    Rectangle {
    id: rr
        color: "orange"
        width: 20;
        height: 20
        y: 40

        Component.onCompleted: SequentialAnimation {
            loops: Animation.Infinite
            running: true
            NumberAnimation { target: rr; property: "x"; to: 180; duration: 600 }
            NumberAnimation { target: rr; property: "x"; to: 0; duration: 600 }
        }
    }

    Component.onCompleted: console.log("ahoj");
}
