#include "benchmark/benchmark.h"

#include <numeric>
#include <random>
#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "SortPoints.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
#include "Point.hpp"
#include "Triangulation.hpp"
#include "GetPixelPoints.hpp"



const char* path = "../../images/bird_l.jpg";
//const int N = 10;

static void CPU_gaussianFilter(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        gaussianFilter(5, 1.4f);
    }
}
BENCHMARK(CPU_gaussianFilter);
//  --------------------------------------------------------------
static void CPU_grayscale(benchmark::State& state)
{
     ced::cpu::Image img(path);
    for(auto _ : state)
    {
        img.convertToGrayscale();
    }
}
BENCHMARK(CPU_grayscale);
//  --------------------------------------------------------------
static void CPU_applyFilter(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    img.convertToGrayscale();
    for(auto _ : state)
    {
        std::vector<float> gfilter = gaussianFilter(5, 1.4f);
        img.applyFilter(gfilter, 5);
    }
    
}
BENCHMARK(CPU_applyFilter);
//  --------------------------------------------------------------
static void CPU_calculateGradients(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
    img.applyFilter(gfilter, gDimension);

    std::vector<float> red   = img.getRedChannel();
    std::vector<float> green = img.getBlueChannel();
    std::vector<float> blue  = img.getGreenChannel();
    for(auto _ : state)
    {
        int height = img.getHeight();
        int width = img.getWidth();
        std::vector<float> orientation(height*width);
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
BENCHMARK(CPU_calculateGradients);
//  --------------------------------------------------------------
static void CPU_nonMaxSup(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
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
BENCHMARK(CPU_nonMaxSup);
//  --------------------------------------------------------------------------------------------------
static void CPU_hysterysis(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
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
BENCHMARK(CPU_hysterysis);
//  --------------------------------------------------------------------------------------------------
static void CPU_getWhitePixels(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
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
    for(auto _ : state)
    {
        getWhitePixelsCoords(white_verts, red, green, blue, height, width);
    }
}
BENCHMARK(CPU_getWhitePixels);
//  --------------------------------------------------------------------------------------------------
static void CPU_triangulateWhitePix(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
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
    int half = white_verts.size()/2;
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin()+10);
    //quickSort(nwhite_verts, 0, nwhite_verts.size());
    for(auto _ : state)
    {
        std::vector<int> triangles;
        triangulate(nwhite_verts, triangles); 
    }
}
BENCHMARK(CPU_triangulateWhitePix);
//  --------------------------------------------------------------------------------------------------
#include "AssignPixToTri.hpp"
static void CPU_assignPixToTri(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    const int gDimension = 5;
    int o_width = img.getWidth(); 
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
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
    int half = white_verts.size()/2;
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin()+10);
    //quickSort(nwhite_verts, 0, nwhite_verts.size());
    std::vector<int> triangles;
    triangulate(nwhite_verts, triangles); 

    for(auto _ : state)
    {
        std::multimap<int, int> pixIDdepTri;
        assignPixToTri(pixIDdepTri, triangles, nwhite_verts, o_width);
    }
}
BENCHMARK(CPU_assignPixToTri);
//  --------------------------------------------------------------------------------------------------
#include "AssignColToPix.hpp"
static void CPU_assignColToPix(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    int o_width = img.getWidth(); 
    std::vector<float> o_red = img.getRedChannel();
    std::vector<float> o_green = img.getGreenChannel();
    std::vector<float> o_blue  = img.getBlueChannel();
    const int gDimension = 5;
    std::vector<float> gfilter = gaussianFilter(gDimension, 1.4f); 
    img.convertToGrayscale();
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
    int half = white_verts.size()/2;
    std::vector<Point> nwhite_verts(white_verts.begin(), white_verts.begin()+10);
    //quickSort(nwhite_verts, 0, nwhite_verts.size());
    std::vector<int> triangles;
    triangulate(nwhite_verts, triangles); 
    std::multimap<int, int> pixIDdepTri;
    int amountOfTri = triangles.size()/3;
    assignPixToTri(pixIDdepTri, triangles, nwhite_verts, o_width);
    for(auto _ : state)
    {
        assignColToPix(o_red, o_green, o_blue, pixIDdepTri, amountOfTri);
    }
}
BENCHMARK(CPU_assignColToPix);

