#include <iostream>
#include <thrust/device_vector.h>
#include "ParamsImageIO.hpp"
int main(int argc, char** argv)
{
    //  ------------------------------------init image------------------------------------
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int num_pixels = width * height;

    // ----------------------------------image -> edge detection--------------------------
    thrust::device_vector<float> red(num_pixels);
    thrust::device_vector<float> green(num_pixels);
    thrust::device_vector<float> blue(num_pixels);
    // -----------------------------------gaussian blur-----------------------------------
    // -----------------------------------gray scale-------------------------------------- 
    // -----------------------------------calculate gradients-----------------------------
    // -----------------------------------nonmaximumSupression----------------------------
    // -----------------------------------get white pixels--------------------------------
    // -----------------------------------triangulation-----------------------------------
    thrust::device_vector<float> x_coords;
    thrust::device_vector<float> y_coords;
    // -----------------------------------colour the triangles---------------------------- 
    // -----------------------------------assign pixel to triangles-----------------------
    // -----------------------------------assign colour to pixels------------------------
    // -----------------------------------save image--------------------------------------
    std::cout<<"Hello"<<std::endl;

    return 0;
}
