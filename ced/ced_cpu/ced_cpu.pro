# lib
TEMPLATE = lib

TARGET = ced_cpu 
QT -= core gui 
OBJECTS_DIR = obj

CONFIG += console c++14
CONFIG -= app_bundle 
#CONFIG += staticlib
DESTDIR = $$PWD/lib

QMAKE_CXXFLAGS += -std=c++14 -fPIC -g -O3 

INCLUDEPATH += \
    /usr/local/include \ 
    $$PWD/include \ 
    /public/devel/2018/include/OpenImageIO

macx:{
    QMAKE_CXXFLAGS += -arch x86_64
    INCLUDEPATH += /usr/local/Cellar/openimageio/
}
LIBS += -L/usr/local/lib  -L/public/devel/2018/lib64 -lOpenImageIO

# Input
HEADERS += $$files(include/*.hpp, true) 
SOURCES += $$files(src/*.cpp,true) 

