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
#include "AssignPixToTri.hpp"
#include "AssignColToPix.hpp"
#include "AvgColour.hpp"
#include "IsPixInTri.hpp"

#include "ParamsImageIO.hpp"

using namespace ced::cpu;
using namespace OIIO; 
int main(int argc, char **argv)
{
    const char *filename    = argv[1];
    int N = std::atoi(argv[2]);
    // read image and store original data for later
    ced::cpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    int o_height = img.getHeight();
    int o_width = img.getWidth(); 
    std::vector<float> o_red = img.getRedChannel();
    std::vector<float> o_green = img.getGreenChannel();
    std::vector<float> o_blue  = img.getBlueChannel();
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = ced::cpu::gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale(); 
    img.saveImage(outgray, true);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    img.saveImage(outgaussian, true);
    // sobel operator to get magnitude and orientation
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    // need to work on this  
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );

    // nonmaximum supression    
    ced::cpu::nonMaximumSupression( height, 
                                    width, 
                                    orientation, 
                                    red,
                                    green,
                                    blue);
    //     final image
    ced::cpu::hysterysis(   red,
                            green,
                            blue, 
                            height, 
                            width, 
                            0.4f, 
                            0.7f);


    // get white pixels
    std::vector<Point> white_verts;
    getWhitePixelsCoords(   white_verts,
                            red,
                            green,
                            blue,
                            height, 
                            width);
    // how many white pixels
    std::random_device rd;
    std::mt19937 k(rd());
    std::shuffle(white_verts.begin(), white_verts.end(), k);
    std::cout<<white_verts.size()<<std::endl;
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin()+N);
    std::cout<<nwhite_verts.size()<<std::endl;

    // generateRandomPoints
    // and add to white points
    std::fill(red.begin(), red.end(), 0);
    std::fill(green.begin(), green.end(), 0);
    std::fill(blue.begin(), blue.end(), 0);
    //std::vector<Point>  rand_verts;
    //generateRandomPoints(rand_verts, 100, o_height, o_width);
    //nwhite_verts.insert(nwhite_verts.end(), rand_verts.begin(), rand_verts.end());
    //std::shuffle(nwhite_verts.begin(), nwhite_verts.end(), k);
    quickSort(nwhite_verts, 0, nwhite_verts.size());
    //  ----------------------------------------------------------------------------
    // show how many points there is  
    for(auto r : nwhite_verts)
    {
        red[(r.x + r.y * width)]    = 1.0f;
        green[(r.x + r.y * width)]  = 1.0f;
        blue[(r.x + r.y * width)]   = 1.0f;
    }
    img.setHeight(height);
    img.setWidth(width);
    img.setRedChannel(red);
    img.setGreenChannel(green);
    img.setBlueChannel(blue);
    img.saveImage(outgradient, true);

    //  ----------------------------------------------------------------------------
    // triangulate
    std::vector<int> triangles;
    triangulate(nwhite_verts, triangles);
    // assign triangle to pixel
    std::multimap<int, int> pixIDdepTri; 
    int amountOfTri = triangles.size()/3;
    assignPixToTri(pixIDdepTri, triangles, nwhite_verts,o_width);
    // -> gpu 
    assignColToPix(o_red, o_green, o_blue, pixIDdepTri, amountOfTri);
    img.setRedChannel(o_red);
    img.setGreenChannel(o_green);
    img.setBlueChannel(o_blue);
    img.setHeight(o_height);
    img.setWidth(o_width);
    img.saveImage(finalout, true);

}
