#include "benchmark/benchmark.h"

#include "../../Demo/include/ParamsImageIO.hpp"
#include "../../ced_cpu/include/Image.hpp"
#include "ConvertToGrayScale.cuh"

#include <thrust/device_vector.h>
#include <vector>
#include <chrono>

static void ConvertToGrayScale(benchmark::State& state)
{
    ced::cpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int width = img.getWidth();
    unsigned int height = img.getHeight();
    unsigned int num_pixels = width * height;
    thrust::device_vector<float> red(num_pixels); 
    thrust::device_vector<float> green(num_pixels); 
    thrust::device_vector<float> blue(num_pixels); 
    for(unsigned int id = 0; id < num_pixels; ++id)
    {
        red[id] = originalPixelData[id * 3 + 0];
        green[id] = originalPixelData[id * 3 + 1];
        blue[id] = originalPixelData[id * 3 + 2];
    }
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        ced::gpu::converToGrayScale(red, green, blue);    
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(ConvertToGrayScale)->UseManualTime();
