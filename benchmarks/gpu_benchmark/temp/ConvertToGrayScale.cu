#include "benchmark/benchmark.h"

#include "../../Demo/include/ParamsImageIO.hpp"
#include "Image.hpp"
#include "ConvertToGrayScale.cuh"

#include <vector>
#include <chrono>

static void ConvertToGrayScale(benchmark::State& state)
{
    ced::gpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int width = img.getWidth();
    unsigned int height = img.getHeight();
    thrust::device_vector<float> red = img.getRedChannel(); 
    thrust::device_vector<float> green = img.getGreenChannel(); 
    thrust::device_vector<float> blue = img.getBlueChannel(); 
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        ced::gpu::convertToGrayScale(red, green, blue);    
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(ConvertToGrayScale)->UseManualTime();
