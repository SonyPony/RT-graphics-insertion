import QtQuick 2.12

Item {
    id: component

    signal showTeamLogos
    signal showSponsors

    /*readonly property string homeSlug: "dukla-praha"
    readonly property string awaySlug: "sokol-telnice"*/

    width: 1280
    height: 720
    //opacity: 0.5

    Connections {
        target: sceneWrapper
        onSignal: {
            if(sponsors.opacity != 0)
                component.showTeamLogos()
            else
                component.showSponsors()
        }
    }

    onShowTeamLogos: SequentialAnimation {
        NumberAnimation { target: sponsors; property: "opacity"; to: 0; duration: 1000 }
        NumberAnimation { target: logosContainer; property: "opacity"; to: 1.; 
        duration: 1000 }
    }

    onShowSponsors: SequentialAnimation {
        NumberAnimation { target: logosContainer; property: "opacity"; to: 0.; duration: 1000 }
        NumberAnimation { target: sponsors; property: "opacity"; to: 1; duration: 1000 }
    }

    Item {  // logos
        id: logosContainer

        anchors.fill: parent
        opacity: 0

        Item {
            width: parent.width / 2
            height: parent.height

            Image {
                height: parent.height * 0.7
                width: height

                mipmap: true
                source: (sceneWrapper.homeSlug) ?"res/mc-logos/" + sceneWrapper.homeSlug + ".png" :""
                fillMode: Image.PreserveAspectFit

                anchors.centerIn: parent
            }
        }

        Item {
            width: parent.width / 2
            height: parent.height

            anchors.right: parent.right

            Image {
                height: parent.height * 0.7
                width: height

                mipmap: true
                source: (sceneWrapper.awaySlug) ?"res/mc-logos/" + sceneWrapper.awaySlug + ".png" :""
                fillMode: Image.PreserveAspectFit
                
                anchors.centerIn: parent
            }
        }
    }

    Image { // sponsors
        id: sponsors

        opacity: 1.
        mipmap: true
        anchors.fill: parent
        source: "res/sponsors_2019_mapping.png"
        fillMode: Image.PreserveAspectFit
    }

    /*Rectangle {
    id: rr
        color: "blue"
        width: 60;
        height: 60
        y: 40
        //anchors.fill: parent

        Component.onCompleted: SequentialAnimation {
            loops: Animation.Infinite
            running: true
            NumberAnimation { target: rr; property: "x"; to: 1280 - rr.width; duration: 1000 }
            NumberAnimation { target: rr; property: "x"; to: 0; duration: 1000 }
        }
    }*/

    Component.onCompleted: console.log("Qml scene loaded.");
}
