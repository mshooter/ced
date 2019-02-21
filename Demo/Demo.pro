TEMPLATE = app
TARGET = Demo 
T -= gui core
OBJECTS_DIR = obj 
CONFIG += console c++14
CONFIG -= app_bundle 

DEPENDPATH += . ../ced/ced_cpu/lib 
INCLUDEPATH += ../ced/ced_cpu/include 

INCLUDEPATH += \
   /usr/local/include \
   /public/devel/2018/include \
   /public/devel/2018/include/OpenImageIO

macx:{
    QMAKE_CXXFLAGS += -arch x86_64
    INCLUDEPATH += /usr/local/Cellar/openimageio/
}
SOURCES += $$files(src/*.cpp,true)

LIBS += -L../ced/ced_cpu/lib -lced_cpu -L/usr/local/lib -L/public/devel/2018/lib64 -lOpenImageIO

QMAKE_CXXFLAGS += -O3 -g -fPIC -std=c++14


