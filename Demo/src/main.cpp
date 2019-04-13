#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Triangulation.hpp"
#include "GetPixelPoints.hpp"
#include "GenGeomImg.hpp"
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
    // original image
    ced::cpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    img.setPixelData(originalPixelData);
    img.saveImage(outgray);
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
    getWhitePixelsCoords(white_verts, magnitude, width);
    std::shuffle(white_verts.begin(), white_verts.end(), std::default_random_engine());
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin()+10);
    // user defines that he only wants a certain number of pixels
    std::vector<unsigned int> triangles;
    std::cout<<"triangulate"<<std::endl;
    for(auto p : nwhite_verts)
    {
        magnitude[(p.x + p.y * width) * 3 + 0] = 1;
        magnitude[(p.x + p.y * width) * 3 + 1] = 1;
        magnitude[(p.x + p.y * width) * 3 + 2] = 1;

    }
    // triangulation is weird
    triangulate(nwhite_verts, triangles);
    genGeomImg(originalPixelData, triangles, nwhite_verts, width, height);

    img.setHeight(height);
    img.setWidth(width);
    img.setPixelData(originalPixelData);
    img.saveImage(outgradient);

}
