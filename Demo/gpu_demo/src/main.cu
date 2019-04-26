#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "ParamsImageIO.hpp"

#include "GaussianFilter.cuh"
#include "Image.hpp"

int main(int argc, char** argv)
{
    //  ------------------------------------init image------------------------------------
    ced::gpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int o_width = img.getWidth();
    unsigned int o_height = img.getHeight();
    // ----------------------------------image -> edge detection--------------------------
    // -----------------------------------gray scale-------------------------------------- 
    img.convertToGrayscale();
    img.saveImage(outgray, true);
    // -----------------------------------gaussian blur-----------------------------------
    std::vector<float> gfilter = ced::gpu::gaussianFilter(5, 1.4f);
    img.applyFilter(gfilter, 5);
    img.saveImage(outgaussian, true);
    // -----------------------------------calculate gradients-----------------------------
    // std::vector<float> kernelX = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    // std::vector<float> kernelY = {-1, -2, -1,
    //                                0,  0,  0,
    //                                1,  2,  1};
    // img.applyFilter(kernelX, 3);
    // img.applyFilter(kernelY, 3);
    // img.saveImage(outgaussian, false);
    // -----------------------------------nonmaximumSupression----------------------------
    // -----------------------------------get white pixels--------------------------------
    // -----------------------------------triangulation-----------------------------------
    // -----------------------------------colour the triangles---------------------------- 
    // -----------------------------------assign pixel to triangles-----------------------
    // -----------------------------------assign colour to pixels------------------------
    // -----------------------------------save image--------------------------------------
    return 0;
}
