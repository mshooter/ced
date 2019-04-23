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
int main()
{
    // read image 
    using namespace OIIO; 
    ced::cpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int o_height = img.getHeight();
    unsigned int o_width = img.getWidth(); 
    img.setPixelData(originalPixelData);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = ced::cpu::gaussianFilter(gDimension, 1.4f); 
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
    std::cout<<white_verts.size()<<std::endl;
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin() + 500);

    // generateRandomPoints
    // and add to white points
    // std::fill(magnitude.begin(), magnitude.end(), 0);
    //std::vector<Point>  rand_verts;
    //generateRandomPoints(rand_verts, nwhite_verts, 500, o_height, o_width);
    //nwhite_verts.insert(nwhite_verts.end(), rand_verts.begin(), rand_verts.end());
    //std::shuffle(nwhite_verts.begin(), nwhite_verts.end(), k);
    //quickSort(nwhite_verts, 0, nwhite_verts.size());
    //  ----------------------------------------------------------------------------
    // show how many points there is  
    for(auto r : nwhite_verts)
    {
        magnitude[(r.x + r.y * width) * 3 + 0] = 1.0f;
        magnitude[(r.x + r.y * width) * 3 + 1] = 1.0f;
        magnitude[(r.x + r.y * width) * 3 + 2] = 1.0f;
    }
    img.setHeight(height);
    img.setWidth(width);
    img.setPixelData(magnitude);
    img.saveImage(finalout);
    //  ----------------------------------------------------------------------------
    // triangulate
    std::vector<unsigned int> triangles;
    triangulate(nwhite_verts, triangles);
        
    
    // assign triangle to pixel
    std::multimap<unsigned int, unsigned int> pixIDdepTri;
    unsigned int amountOfTri = triangles.size()/3;
    assignPixToTri(pixIDdepTri, triangles, nwhite_verts, o_height, o_width);
    assignColToPix(originalPixelData, pixIDdepTri, amountOfTri);
    img.setHeight(o_height);
    img.setWidth(o_width);
    img.setPixelData(originalPixelData);
    img.saveImage(outgradient);

}
