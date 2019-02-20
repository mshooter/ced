#include <iostream>
#include <OpenImageIO/imageio.h>
#include <cmath>
#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"

int main()
{
    // read image 
    using namespace OIIO; 
    // std::unique_ptr<ImageInput>
    // std::unique_ptr<ImageOutput>
    #ifdef __APPLE__
        const char *filename = "/Users/moirashooter/Desktop/cat.jpg";
        const char *outfile = "/Users/moirashooter/Desktop/ncat.jpg";
        const char *outgray = "/Users/moirashooter/Desktop/graycat.jpg";
        const char *outgaussian = "/Users/moirashooter/Desktop/gaussiancat.jpg";
        const char *outgradient = "/Users/moirashooter/Desktop/gradientcat.jpg";
    #else
        const char *filename = "/home/s4928793/Desktop/cat.jpg";
        const char *outfile = "/home/s4928793/Desktop/ncat.jpg";
        const char *outgray = "/home/s4928793/Desktop/graycat.jpg";
        const char *outgaussian = "/home/s4928793/Desktop/gaussiancat.jpg";
        const char *outgradient = "/home/s4928793/Desktop/gradientcat.jpg";
    #endif
    ced::Image img(filename);
    // create filter gaussian blur
    int gDimension = 5;
    std::vector<float> gfilter = ced::gaussianFilter(gDimension, 2); 
    // convert to gray scale
    img.convertToGrayscale();
    img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    img.saveImage(outgaussian);
    // sobel operator to get magnitude and orientation
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> magnitude = img.getPixelData();
    // mag = pixelData, orientations 
    ced::calculateGradients(height,
                            width, 
                            magnitude,
                            orientation
                            );
    magnitude.resize(height*width*3);
    img.setHeight(height);
    img.setWidth(width);
    img.setPixelData(magnitude);
    img.saveImage(outgradient);
    // nonmaximum supression    
    ced::nonMaximumSupression(height, width, orientation, magnitude);


    
    return 0;
}
