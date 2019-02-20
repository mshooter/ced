#include <iostream>
#include <OpenImageIO/imageio.h>
#include <cmath>
#include "GaussianFilter.hpp"
#include "Gradients.hpp"
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
        const char *outfile = "/home/s4928793/Desktop/ncat.jpg";
    #endif
    ced::Image img(filename);
    // create filter gaussian blur
    int gDimension = 5;
    std::vector<float> gfilter = ced::gaussianFilter(gDimension, 2); 
    // convert to gray scale
    img.convertToGrayscale();
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    // sobel operator to get magnitude and orientation
    std::vector<float> magnitudes; 
    std::vector<float> orientation;
    int height = img.getHeight();
    int width = img.getWidth();
    
    // initialise image
    // setPixelData
    // setHeight
    // set Width

    // write image
    img.saveImage(outfile);
        


    
    return 0;
}
