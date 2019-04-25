#include "benchmark/benchmark.h"
#include "GaussianFilter.cuh"
#include "Image.hpp"
#include "ImageApplyFilter.cuh"
#include <thrust/device_vector.h>
#include "../../Demo/include/ParamsImageIO.hpp"

using namespace ced::gpu;
static void applyFilterKernel(benchmark::State& state)
{
    ced::gpu::Image img(filename);
    std::vector<float> m_pixelData = img.getPixelData();
    int _dimension = 5;
    unsigned int m_width = img.getWidth();
    unsigned int m_height = img.getHeight();
    std::vector<float> _filter = ced::gpu::gaussianFilter(_dimension, 1.4f);
    // ----------------------host allocation----------------------------------
    // we have the filter 
    // we have the original image
    int nwidth = m_width - _dimension  + 1;
    int nheight = m_height - _dimension + 1;
    // --------------------device allocation----------------------------------
    thrust::device_vector<float> d_oimage = m_pixelData;
    thrust::device_vector<float> d_nimage(nheight * nwidth * 3);
    thrust::device_vector<float> d_filter = _filter;            
    // --------------------typecast raw ptr-----------------------------------
    float* d_oimage_ptr = thrust::raw_pointer_cast(d_oimage.data());
    float* d_nimage_ptr = thrust::raw_pointer_cast(d_nimage.data());
    float* d_filter_ptr = thrust::raw_pointer_cast(d_filter.data());
    // --------------------execution config-----------------------------------
    int blockW = 32;
    int blockH = 32;
    const dim3 grid(iDivUp(nwidth, blockW),
                    iDivUp(nheight, blockH));
    const dim3 threadBlock(blockW, blockH);
    // --------------------calling kernel-------------------------------------
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        d_applyFilter<<<grid, threadBlock>>>(d_oimage_ptr, 
                                             d_nimage_ptr,   
                                             d_filter_ptr,
                                             nwidth,
                                             nheight, 
                                             _dimension);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());

    }
}
BENCHMARK(applyFilterKernel)->UseManualTime();
 
