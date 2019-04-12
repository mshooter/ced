#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Triangulation.hpp"
#include "GetPixelPoints.hpp" 
#include <numeric>
#include <random>
int main()
{
    // read image 
    using namespace OIIO; 
    // std::unique_ptr<ImageInput>
    // std::unique_ptr<ImageOutput>
    #ifdef __APPLE__
        const char *filename    = "/Users/moirashooter/Desktop/Cat/baby.jpg";
        const char *outgray     = "/Users/moirashooter/Desktop/Cat/graycat.jpg";
        const char *outgaussian = "/Users/moirashooter/Desktop/Cat/gaussiancat.jpg";
        const char *outgradient = "/Users/moirashooter/Desktop/Cat/nonMaximumSupressioncat.jpg";
        const char *finalout    = "/Users/moirashooter/Desktop/Cat/finalcat.jpg";
    #else
        const char *filename    = "/home/s4928793/Desktop/Cat/cat.jpg";
        const char *outgray     = "/home/s4928793/Desktop/Cat/graycat.jpg";
        const char *outgaussian = "/home/s4928793/Desktop/Cat/gaussiancat.jpg";
        const char *outgradient = "/home/s4928793/Desktop/Cat/nonMaximumSupressioncat.jpg";
        const char *finalout    = "/home/s4928793/Desktop/Cat/finalcat.jpg";
    #endif
    ced::cpu::Image img(filename);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = ced::cpu::gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    //img.saveImage(outgaussian);
    // sobel operator to get magnitude and orientation
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> magnitude = img.getPixelData();
    // need to work on this  
    ced::cpu::calculateGradients(height,
                          width, 
                          magnitude,
                          orientation
                          );
    // nonmaximum supression    
    ced::cpu::nonMaximumSupression(height, width, orientation, magnitude);
    img.setHeight(height);
    img.setWidth(width);
    img.setPixelData(magnitude);
    img.saveImage(outgradient);
    //     final image
    ced::cpu::hysterysis(magnitude, height, width, 0.2f, 0.3f);
    img.setPixelData(magnitude);

    using namespace ced::cpu;
    // all white pixels
    std::vector<Point> white_verts;
    std::vector<Point> original_verts;
    // get the original points
    getWhitePixelsCoords(white_verts, original_verts, magnitude, height, width);
    // user defines that he only wants a certain number of pixels
    std::vector<unsigned int> triangles;
    std::cout<<"triangulate"<<std::endl;
    //triangulate(white_verts, triangles);
    std::fill(magnitude.begin(), magnitude.end(), 0);
    
    for(int i=0; i < 2000 ; ++i)
    {
        Point v = original_verts[i];
        magnitude[(v.x + v.y * width) * 3 ] = 1;
        magnitude[(v.x + v.y * width) * 3 + 1] = 1;
        magnitude[(v.x + v.y * width) * 3 + 2] = 1;

    }

    img.setHeight(height);
    img.setWidth(width);
    img.setPixelData(magnitude);
    img.saveImage(outgradient);

}
