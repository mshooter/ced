TEMPLATE = app
TARGET = cpu_tests

QT-= core gui 

OBJECTS_DIR = obj 

CONFIG += console c++14
CONFIG -= app_bundle

SOURCES += $$files(src/*.cpp,true)
HEADERS += $$files(include/*(.hpp|.inl), true) 

DEPENDPATH += . ../../ced_cpu 
INCLUDEPATH += ../../ced_cpu/include 
INCLUDEPATH += \
   /usr/local/include/gtest \ 
   /usr/local/include \ 
   $$PWD/include  
  
QMAKE_CXXFLAGS += -std=c++14 -fPIC -g -O3 

LIBS += -L../../ced_cpu/lib -lced_cpu -L/usr/local/lib -lgtest -pthread

