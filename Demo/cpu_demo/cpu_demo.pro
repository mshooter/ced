TEMPLATE = app
TARGET = cpu_demo 
T -= gui core
OBJECTS_DIR = obj 
CONFIG += console c++14
CONFIG -= app_bundle 

DEPENDPATH += . ../../ced_cpu/lib 
INCLUDEPATH += ../../ced_cpu/include 

INCLUDEPATH += \
    ../include\
   $$PWD/include \ 
   /usr/local/include \
   /public/devel/2018/include \
   /public/devel/2018/include/OpenImageIO

macx:{
    QMAKE_CXXFLAGS += -arch x86_64
    INCLUDEPATH += /usr/local/Cellar/openimageio/
}
SOURCES += $$files(src/*.cpp,true)
HEADERS += $$files(include/*.hpp,true)

LIBS += -L../../ced_cpu/lib -lced_cpu -L/usr/local/lib -L/public/devel/2018/lib64 -lOpenImageIO 

# -pg
QMAKE_CXXFLAGS += -O3 -g -fPIC -std=c++14
#QMAKE_LFLAGS += -pg

