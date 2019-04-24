#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "ParamsImageIO.hpp"

#include "ConvertToGrayScale.cuh"
#include "GaussianFilter.cuh"
#include "Image.hpp"

int main(int argc, char** argv)
{
    //  ------------------------------------init image------------------------------------
    ced::gpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int width = img.getWidth();
    unsigned int height = img.getHeight();
    unsigned int num_pixels = width * height;
    img.convertToGrayscale();
    img.saveImage(outgray, true);
    // ----------------------------------image -> edge detection--------------------------
    // -----------------------------------gaussian blur-----------------------------------
    std::vector<float> gfilter = ced::gpu::gaussianFilter(5, 1.4f);
    img.applyFilter(gfilter, 5);
    //img.saveImage(outgaussian, false);
    std::vector<float> kernelX = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    std::vector<float> kernelY = {-1, -2, -1,
                                   0,  0,  0,
                                   1,  2,  1};
    img.applyFilter(kernelX, 3);
    img.applyFilter(kernelY, 3);
    img.saveImage(outgaussian, false);
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
