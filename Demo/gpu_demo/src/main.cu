#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "ParamsImageIO.hpp"
#include "Gradients.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Point.hpp"
#include "GetPixelPoints.hpp"

#include "Image.hpp"
#include "GaussianFilter.cuh"


int main(int argc, char** argv)
{
    //  ------------------------------------init image------------------------------------
    ced::gpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int o_width = img.getWidth();
    unsigned int o_height = img.getHeight();
    thrust::device_vector<float> o_red = img.getRedChannel();
    thrust::device_vector<float> o_green = img.getGreenChannel();
    thrust::device_vector<float> o_blue  = img.getBlueChannel();
    // ----------------------------------image -> edge detection--------------------------
    // -----------------------------------gray scale-------------------------------------- 
    img.convertToGrayscale();
    img.saveImage(outgray, true);
    // -----------------------------------gaussian blur-----------------------------------
    std::vector<float> gfilter = ced::gpu::gaussianFilter(5, 1.4f);
    img.applyFilter(gfilter, 5);
    img.saveImage(outgaussian, true);
    // -----------------------------------calculate gradients-----------------------------
    // std::vector<float> kernelX = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    // std::vector<float> kernelY = {-1, -2, -1,
    //                                0,  0,  0,
    //                                1,  2,  1};
    int width = img.getWidth();
    int height = img.getHeight();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(width, height, red, green, blue, orientation);
    ced::cpu::nonMaximumSupression( height, 
                                    width, 
                                    orientation, 
                                    red,
                                    green,
                                    blue);
    ced::cpu::hysterysis(   red,
                            green,
                            blue, 
                            height, 
                            width, 
                            0.4f, 
                            0.7f);
    std::vector<Point> white_verts;
    ced::cpu::getWhitePixelsCoords(   white_verts,
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
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin()+10);
    std::vector<unsigned int> triangles;
    ced::cpu::triangulate(nwhite_verts, triangles);
    std::multimap<unsigned int, unsigned int> pixIDdepTri; 
    unsigned int amountOfTri = triangles.size()/3;
    ced::cpu::assignPixToTri(pixIDdepTri, triangles, nwhite_verts, o_height, o_width);
    // assign
    std::vector<int> h_triID;
    std::vector<int> h_PixID;
    for(auto x : pixIDdepTri)
    {
        h_triID.push_back(x.first);
        h_pixID.push_back(x.second);
    }
    // do the gpu thing
    thrust::device_vector<float> d_triID = h_triID;
    thrust::device_vector<float> d_PixID = h_PixID;
    ced::gpu::assignColToPix(o_red, o_green, o_blue, d_triID, d_PixID, (int)amountOfTri);
    // -----------------------------------nonmaximumSupression----------------------------
    // -----------------------------------get white pixels--------------------------------
    // -----------------------------------triangulation-----------------------------------
    // -----------------------------------colour the triangles---------------------------- 
    // -----------------------------------assign pixel to triangles-----------------------
    // -----------------------------------assign colour to pixels------------------------
    // -----------------------------------save image--------------------------------------
    return 0;
}
