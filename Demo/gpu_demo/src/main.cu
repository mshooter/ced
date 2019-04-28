#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "ParamsImageIO.hpp"
#include "Gradients.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Point.hpp"
#include "GetPixelPoints.hpp"

#include "Image.cuh"
#include "GaussianFilter.cuh"


int main(int argc, char** argv)
{
    const char* filename = argv[1];
    const char* outgray = argv[2];
    const char* outgaussian = argv[3];
    //  ------------------------------------init image------------------------------------
    ced::gpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int o_width = img.getWidth();
    unsigned int o_height = img.getHeight();
    thrust::device_vector<float> o_red = img.getRedChannel();
    thrust::device_vector<float> o_green = img.getGreenChannel();
    thrust::device_vector<float> o_blue  = img.getBlueChannel();
    // ----------------------------------image -> edge detection--------------------------
    // -----------------------------------gray scale-------------------------------------- 
    img.convertToGrayscale();
    img.saveImage(outgray, true);
    // -----------------------------------gaussian blur-----------------------------------
    std::vector<float> gfilter = ced::gpu::gaussianFilter(5, 1.4f);
    img.applyFilter(gfilter, 5);
    img.saveImage(outgaussian, true);
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
