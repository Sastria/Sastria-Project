#-------------------------------------------------
#
# Project created by QtCreator 2013-09-18T11:31:44
#
#-------------------------------------------------

QT       += core gui multimedia multimediawidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SaStriaTool
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    threadtraker.cpp \
    videoresults.cpp

HEADERS  += mainwindow.h \
    threadtraker.h \
    videoresults.h

FORMS    += mainwindow.ui \
    videoresults.ui

