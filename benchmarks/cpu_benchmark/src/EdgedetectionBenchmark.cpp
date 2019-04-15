#include "benchmark/benchmark.h"

#include "GaussianFilter.hpp"
#include "Gradients.hpp"
#include "Image.hpp"
#include "NonMaximumSupression.hpp"
#include "Hysterysis.hpp"
const char* linuxPath = "/home/s4928793/Desktop/Cat/cat.jpg";
const char* path = "/Users/moirashooter/Desktop/Cat/catt.jpg";

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
    Image img(path);
    for(auto _ : state)
    {
        img.convertToGrayscale();
    }
}
BENCHMARK(BM_grayscale);
//  --------------------------------------------------------------
static void BM_applyFilter(benchmark::State& state)
{
    using namespace ced::cpu;
    Image img(path);
    img.convertToGrayscale();
    std::vector<float> gfilter = gaussianFilter(5, 1.4f);
    for(auto _ : state)
    {
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
    //img.saveImage(outgaussian);
    // sobel operator to get magnitude and orientation
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> magnitude = img.getPixelData();
    // need to work on this  
    for(auto _ : state)
    {
        calculateGradients(height,
                           width, 
                           magnitude,
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
    //img.saveImage(outgaussian);
    // sobel operator to get magnitude and orientation
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> magnitude = img.getPixelData();
    calculateGradients(height,
                       width, 
                       magnitude,
                       orientation
                       );
    for(auto _ : state)
    {
        ced::cpu::nonMaximumSupression(height, width, orientation, magnitude);
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
    //img.saveImage(outgaussian);
    // sobel operator to get magnitude and orientation
    int height = img.getHeight();
    int width = img.getWidth();
    std::vector<float> orientation;
    std::vector<float> magnitude = img.getPixelData();
    calculateGradients(height,
                       width, 
                       magnitude,
                       orientation
                       );
    nonMaximumSupression(height, width, orientation, magnitude);
    for(auto _ : state)
    {
        hysterysis(magnitude, height, width, 0.2f, 0.3f);
    }
}
BENCHMARK(BM_hysterysis);

