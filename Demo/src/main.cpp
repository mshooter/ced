#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Point.hpp"
#include "DrawLine.hpp"
#include "GenerateRandomPoints.hpp"
#include "SortPoints.hpp"
#include "SplitVector.hpp"

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
    // get random points
    std::vector<ced::cpu::Point> coord = ced::cpu::generateRandomPoints(100, img.getHeight(), img.getWidth());
    // sort the points
    ced::cpu::quickSort(coord, 0, coord.size()-1);
    // img data 
    std::vector<float> imgData(img.getHeight() * img.getWidth() * 3, 0.0f);
    // split the vector 
    std::vector<std::vector<ced::cpu::Point>> fvec;
    ced::cpu::splitVector(fvec, coord);
    for(auto x : fvec)
    {
        int s = x.size();
        if(s == 2)
        {
                imgData[(x[0].getX() + x[0].getY() * img.getWidth()) * 3 + 0] = 1;
                imgData[(x[0].getX() + x[0].getY() * img.getWidth()) * 3 + 1] = 1;
                imgData[(x[0].getX() + x[0].getY() * img.getWidth()) * 3 + 2] = 1;

                imgData[(x[1].getX() + x[1].getY() * img.getWidth()) * 3 + 0] = 1;
                imgData[(x[1].getX() + x[1].getY() * img.getWidth()) * 3 + 1] = 1;
                imgData[(x[1].getX() + x[1].getY() * img.getWidth()) * 3 + 2] = 1;
                //ced::cpu::drawLine(x[0], x[1], imgData, img.getWidth());

        }

    }
    // draw the points

    // draw the lines now


    // draw the lines
    img.setPixelData(imgData);
    img.saveImage(finalout);
    //// create filter gaussian blur
    //const int gDimension = 5;
    //std::vector<float> gfilter = ced::cpu::gaussianFilter(gDimension, 1.4f); 
    //// convert to gray scale
    //img.convertToGrayscale();
    ////img.saveImage(outgray);
    //// apply gaussian filter
    //img.applyFilter(gfilter, gDimension);
    ////img.saveImage(outgaussian);
    //// sobel operator to get magnitude and orientation
    //int height = img.getHeight();
    //int width = img.getWidth();
    //std::vector<float> orientation;
    //std::vector<float> magnitude = img.getPixelData();
    //// need to work on this  
    //ced::cpu::calculateGradients(height,
    //                      width, 
    //                      magnitude,
    //                      orientation
    //                      );
    //// nonmaximum supression    
    //ced::cpu::nonMaximumSupression(height, width, orientation, magnitude);
    //img.setHeight(height);
    //img.setWidth(width);
    //img.setPixelData(magnitude);
    //img.saveImage(outgradient);
    // final image
    //ced::cpu::hysterysis(magnitude, height, width, 0.2f, 0.3f);
    //img.setPixelData(magnitude);
    //img.saveImage(finalout);
    

    
    return 0;
}
