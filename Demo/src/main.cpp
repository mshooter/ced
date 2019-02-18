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
    // apply gaussian filter
    img.applyFilter(gfilter, dimension);
    // sobel edge detector 
    std::vector<int> kernelX = {-1, 0, 1,
                                -2, 0, 2,
                                -1, 0, 1};
    std::vector<int> kernelY = {-1, -2, -1,
                                 0,  0,  0,
                                -1, -2, -1};

    // write image
    img.saveImage(outfile);
        


    
    return 0;
}
