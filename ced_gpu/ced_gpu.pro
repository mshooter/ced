# dynamic lib
TEMPLATE = lib

# installation directory for library
TARGET = ced_gpu 

QT -= core gui 

# Directory to store all the intermediate objects
OBJECTS_DIR = obj

# Use the following path for nvcc created object files
CUDA_OBJECTS_DIR = cudaobj

CONFIG += console c++11
CONFIG -= app_bundle 
DESTDIR = $$PWD/lib

QMAKE_CXXFLAGS += -std=c++11 -fPIC -g -O3 

INCLUDEPATH += \
    /usr/local/include \ 
    $$PWD/include \ 
    /public/devel/2018/include/OpenImageIO

LIBS += -L/usr/local/lib  -L/public/devel/2018/lib64 -lOpenImageIO

HEADERS += $$files(include/*(.hpp | cuh), true) 
CUDA_SOURCES += $$files(src/*.cu,true) 

# CUDA stuff
# link with following libraries
LIBS += $$system(pkg-config --silence-errors --libs cuda-8.0 cudart-8.0 curand-8.0 cublas-8.0) -lcublas_device -lcudadevrt

# Directories
INCLUDEPATH += \
    ${CUDA_PATH}/include \
    ${CUDA_PATH}/include/cuda

# Set this environment variable yourself.
CUDA_DIR=${CUDA_PATH}
isEmpty(CUDA_DIR) {
    message(CUDA_DIR not set - set this to the base directory of your local CUDA install (on the labs this should be /usr))
}

## CUDA_INC - all includes needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC += $$join(INCLUDEPATH,' -I','-I',' ')

## nvcc flags 
CUDA_COMPUTE_ARCH=${CUDA_ARCH}
isEmpty(CUDA_COMPUTE_ARCH) {
    message(CUDA_COMPUTE_ARCH environment variable not set - set this to your local CUDA compute capability.)
}

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = -ccbin ${HOST_COMPILER} -g -O3 -pg -gencode arch=compute_30,code=sm_30 --compiler-options -fno-strict-aliasing --compiler-options -fPIC --std=c++11 -use_fast_math  #--ptxas-options=-v

# Define the path and binary for nvcc
NVCCBIN = $$CUDA_DIR/bin/nvcc

# prepare intermediate cuda compiler
cuda.input  = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o 
cuda.commands = $$NVCCBIN $$NVCCFLAGS -dc $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} - o ${QMAKE_FILE_OUT}

# set our variable out, these obj files need to be used to create the link obj file and used in our final gcc compilation
cuda.variable_out      = CUDA_OBJ
cuda.variable_out     += OBJECTS 
cuda.clean             = $$CUDA_OBJECTS_DIR/*.o
cuda.CONFIG            = no_link
QMAKE_EXTRA_COMPILERS += cuda

# prepare the linking compiler step 
cudalink.input  = CUDA_OBJ
cudalink.CONFIG = combine
cudalink.output = $$OBJECTS_DIR/cuda_link.o

# tweak arch according to your hws compute capability 
cudalink.commands = $$NVCCBIN $$NVCCFLAGS $$CUDA_INC -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia -lcuda -lcudart -lcudadevrt -lcurand
cudalink.dependency_type = TYPE_C
cudalink.depend_command = $$NVCCBIN $$NVCCFLAGS -M $$CUDA_INC ${QMAKE_FILE_NAME}

# tell qt that we want add more stuff to makefile 
QMAKE_EXTRA_COMPILERS += cudalink

