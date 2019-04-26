#include "benchmark/benchmark.h"

#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Point.hpp"
#include "Triangulation.hpp"
#include "GetPixelPoints.hpp"
#include <numeric>
#include <random>

#ifdef __APPLE__
    const char *path    = "/Users/moirashooter/Desktop/Cat/bob.jpg";
    const char *outgray     = "/Users/moirashooter/Desktop/Cat/graycat.jpg";
    const char *outgaussian = "/Users/moirashooter/Desktop/Cat/gaussiancat.jpg";
    const char *outgradient = "/Users/moirashooter/Desktop/Cat/nonMaximumSupressioncat.jpg";
    const char *finalout    = "/Users/moirashooter/Desktop/Cat/edge_image.jpg";
#else
    const char *path   = "/home/s4928793/Desktop/Cat/logo.jpg";
    const char *outgray     = "/home/s4928793/Desktop/Cat/graycat.jpg";
    const char *outgaussian = "/home/s4928793/Desktop/Cat/gaussiancat.jpg";
    const char *outgradient = "/home/s4928793/Desktop/Cat/nonMaximumSupressioncat.jpg";
    const char *finalout    = "/home/s4928793/Desktop/Cat/edge_image.jpg";
#endif

static void BM_gaussianFilter(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        gaussianFilter(5, 1.4f);
    }
}
BENCHMARK(BM_gaussianFilter);
//  --------------------------------------------------------------
static void BM_grayscale(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        Image img(path);
        img.convertToGrayscale();
    }
}
BENCHMARK(BM_grayscale);
//  --------------------------------------------------------------
static void BM_applyFilter(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        Image img(path);
        img.convertToGrayscale();
        std::vector<float> gfilter = gaussianFilter(5, 1.4f);
        img.applyFilter(gfilter, 5);
    }
    
}
BENCHMARK(BM_applyFilter);
//  --------------------------------------------------------------
static void BM_calculateGradients(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);

    for(auto _ : state)
    {
        int height = img.getHeight();
        int width = img.getWidth();
        std::vector<float> orientation;
        std::vector<float> red   = img.getRedChannel();
        std::vector<float> green = img.getBlueChannel();
        std::vector<float> blue  = img.getGreenChannel();
        ced::cpu::calculateGradients(
                              height,
                              width, 
                              red, 
                              green, 
                              blue,
                              orientation
                              );
    }
}
BENCHMARK(BM_calculateGradients);
//  --------------------------------------------------------------
static void BM_nonMaxSup(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );
    for(auto _ : state)
    {
        ced::cpu::nonMaximumSupression( height, 
                                        width, 
                                        orientation, 
                                        red,
                                        green,
                                        blue);
    }
}
BENCHMARK(BM_nonMaxSup);
//  --------------------------------------------------------------------------------------------------
static void BM_hysterysis(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );
    ced::cpu::nonMaximumSupression( height, 
                                    width, 
                                    orientation, 
                                    red,
                                    green,
                                    blue);
    for(auto _ : state)
    {
   
        ced::cpu::hysterysis(   red,
                                green,
                                blue, 
                                height, 
                                width, 
                                0.4f, 
                                0.7f);
 
    }
}
BENCHMARK(BM_hysterysis);
//  --------------------------------------------------------------------------------------------------
static void BM_getWhitePixels(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );
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

    for(auto _ : state)
    {
        std::vector<Point> white_verts; 
        std::random_device rd;
        std::mt19937 k(rd());
        std::shuffle(white_verts.begin(), white_verts.end(), k);
        getWhitePixelsCoords(white_verts, red, green, blue, height, width);
    }
}
BENCHMARK(BM_getWhitePixels);
//  --------------------------------------------------------------------------------------------------
static void BM_triangulateWhitePix(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );
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
    std::random_device rd;
    std::mt19937 k(rd());
    std::shuffle(white_verts.begin(), white_verts.end(), k);
    getWhitePixelsCoords(white_verts, red, green, blue, height, width);
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.end());
    for(auto _ : state)
    {
        std::vector<unsigned int> triangles;
        triangulate(nwhite_verts, triangles); 
    }
}
BENCHMARK(BM_triangulateWhitePix);
//  --------------------------------------------------------------------------------------------------
#include "AssignPixToTri.hpp"
static void BM_assignPixToTri(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    unsigned int o_height = img.getHeight();
    unsigned int o_width = img.getWidth(); 
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );
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
    std::random_device rd;
    std::mt19937 k(rd());
    std::shuffle(white_verts.begin(), white_verts.end(), k);
    getWhitePixelsCoords(white_verts, red, green, blue, height, width);
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.end());
    std::vector<unsigned int> triangles;
    triangulate(nwhite_verts, triangles); 

    for(auto _ : state)
    {
//       std::multimap<unsigned int, unsigned int> pixIDdepTri;
//       assignPixToTri(pixIDdepTri, triangles, nwhite_verts, o_height, o_width);
    }
}
BENCHMARK(BM_assignPixToTri);
//  --------------------------------------------------------------------------------------------------
#include "AssignColToPix.hpp"
static void BM_assignColToPix(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    unsigned int o_height = img.getHeight();
    unsigned int o_width = img.getWidth(); 
    std::vector<float> o_red = img.getRedChannel();
    std::vector<float> o_green = img.getGreenChannel();
    std::vector<float> o_blue  = img.getBlueChannel();
    // create filter gaussian blur
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    // convert to gray scale
    img.convertToGrayscale();
    //img.saveImage(outgray);
    // apply gaussian filter
    img.applyFilter(gfilter, gDimension);
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    ced::cpu::calculateGradients(
                          height,
                          width, 
                          red, 
                          green, 
                          blue,
                          orientation
                          );
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
    std::random_device rd;
    std::mt19937 k(rd());
    std::shuffle(white_verts.begin(), white_verts.end(), k);
    getWhitePixelsCoords(white_verts, red, green, blue, height, width);
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.end());
    std::vector<unsigned int> triangles;
    triangulate(nwhite_verts, triangles); 
    std::multimap<unsigned int, unsigned int> pixIDdepTri;
    unsigned int amountOfTri = triangles.size()/3;
    //assignPixToTri(pixIDdepTri, triangles, nwhite_verts, o_height, o_width);
    for(auto _ : state)
    {
        //assignColToPix(o_red, o_green, o_blue, pixIDdepTri, amountOfTri);
    }
}
BENCHMARK(BM_assignColToPix);

