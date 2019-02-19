#include <iostream>
#include <OpenImageIO/imageio.h>
#include <cmath>
#include "GaussianFilter.hpp"
#include "Image.hpp"
int main()
{
    // read image 
    using namespace OIIO; 
    // std::unique_ptr<ImageInput>
    // std::unique_ptr<ImageOutput>
    #ifdef __APPLE__
        const char *filename = "/Users/moirashooter/Desktop/cat.jpg";
        const char *outfile = "/Users/moirashooter/Desktop/ncat.jpg";
    #else
        const char *filename = "/home/s4928793/Desktop/cat.jpg";
        const char *outfile = "/home/s4928793/Desktop/bobcat.jpg";
    #endif
    ced::Image img(filename);
    // create filter gaussian blur
    int dimension = 5; 
    std::vector<float> gfilter = ced::gaussianFilter(dimension, 2); 
    // convert to gray scale
    img.convertToGrayscale();
    // apply gaussian filter
    img.applyFilter(gfilter, dimension);
    img.applySobelFilter();

    
    // write image
    img.saveImage(outfile);
        


    
    return 0;
}
