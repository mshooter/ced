#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "ParamsImageIO.hpp"
#include "../../ced_cpu/include/Image.hpp"

#include "ConvertToGrayScale.cuh"

int main(int argc, char** argv)
{
    //  ------------------------------------init image------------------------------------
    ced::cpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int width = img.getWidth();
    unsigned int height = img.getHeight();
    unsigned int num_pixels = width * height;
    thrust::device_vector<float> red(num_pixels); 
    thrust::device_vector<float> green(num_pixels); 
    thrust::device_vector<float> blue(num_pixels); 
    for(unsigned int id = 0; id < num_pixels; ++id)
    {
        red[id] = originalPixelData[id * 3 + 0];
        green[id] = originalPixelData[id * 3 + 1];
        blue[id] = originalPixelData[id * 3 + 2];
    }
    ced::gpu::converToGrayScale(red, green, blue);    
    std::vector<float> h_red(num_pixels);
    std::vector<float> h_green(num_pixels);
    std::vector<float> h_blue(num_pixels);
    thrust::copy(red.begin(), red.end(), h_red.begin());    
    thrust::copy(green.begin(), green.end(), h_green.begin());    
    thrust::copy(blue.begin(), blue.end(), h_blue.begin());    
    for(unsigned int id = 0; id < num_pixels; ++id)
    {
        originalPixelData[id * 3 + 0] = h_red[id];
        originalPixelData[id * 3 + 1] = h_green[id];
        originalPixelData[id * 3 + 2] = h_blue[id];
    }
    img.setPixelData(originalPixelData);
    img.saveImage(outgray);
    // ----------------------------------image -> edge detection--------------------------
    // -----------------------------------gaussian blur-----------------------------------
    // -----------------------------------gray scale-------------------------------------- 
    // -----------------------------------calculate gradients-----------------------------
    // -----------------------------------nonmaximumSupression----------------------------
    // -----------------------------------get white pixels--------------------------------
    // -----------------------------------triangulation-----------------------------------
    // -----------------------------------colour the triangles---------------------------- 
    // -----------------------------------assign pixel to triangles-----------------------
    // -----------------------------------assign colour to pixels------------------------
    // -----------------------------------save image--------------------------------------
    return 0;
}
