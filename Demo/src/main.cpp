#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Triangulation.hpp"
#include "GetPixelPoints.hpp"
#include "Distance2P.hpp"
#include "TriOrientation.hpp"
#include "SortPoints.hpp"
#include <numeric>
#include <random>
#include "GenerateRandomPoints.hpp"
#include "AssignTriToPix.hpp"
#include "AssignColToPix.hpp"

using namespace ced::cpu;
int main()
{
    // read image 
    using namespace OIIO; 
    // std::unique_ptr<ImageInput>
    // std::unique_ptr<ImageOutput>
    #ifdef __APPLE__
        const char *filename    = "/Users/moirashooter/Desktop/Cat/bob.jpg";
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
    unsigned int o_height = img.getHeight();
    unsigned int o_width = img.getWidth(); 
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
    //     final image
    ced::cpu::hysterysis(magnitude, height, width, 0.4f, 0.5f);
    // get white pixels
    std::vector<Point> white_verts;
    getWhitePixelsCoords(white_verts, magnitude, height, width);
    // how many white pixels
    std::random_device rd;
    std::mt19937 k(rd());
    std::shuffle(white_verts.begin(), white_verts.end(), k);
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.end());
    std::fill(magnitude.begin(), magnitude.end(), 0);
    std::vector<Point>  rand_verts;
    // generateRandomPoints
    // and add to white points
    generateRandomPoints(rand_verts, nwhite_verts, 100, height, width);
    nwhite_verts.insert(nwhite_verts.end(), rand_verts.begin(), rand_verts.end());
    quickSort(nwhite_verts, 0, nwhite_verts.size());
    // show how many points there is  
   //for(auto r : nwhite_verts)
   //{
   //    magnitude[(r.x + r.y * width) * 3 + 0] = 1.0f;
   //    magnitude[(r.x + r.y * width) * 3 + 1] = 1.0f;
   //    magnitude[(r.x + r.y * width) * 3 + 2] = 1.0f;
   //}
   //img.setHeight(height);
   //img.setWidth(width);
   //img.setPixelData(magnitude);
   //img.saveImage(finalout);
    // triangulate
    std::vector<unsigned int> triangles;
    triangulate(nwhite_verts, triangles);
    // assign triangle to pixel
    std::vector<Point> ptIdx; 
    std::vector<unsigned int> triangle; 
    assignTriToPix(triangle, ptIdx, triangles, nwhite_verts, height, width);
    // assign colour to pixels depending on triangle 
    std::cout<<triangles.size()/3<<std::endl;
    int currtt = 0;
    int oldtt = 0;
    // creates triangles
    assignColToPix(originalPixelData, triangles.size()/3, triangle, ptIdx, o_height, o_width);
    // output
    img.setHeight(o_height);
    img.setWidth(o_width);
    img.setPixelData(originalPixelData);
    img.saveImage(outgradient);

}
