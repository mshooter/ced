#include "benchmark/benchmark.h"

#include "GaussianFilter.cuh"
#include "Image.hpp"
#include "ImageApplyFilter.cuh"
#include "ThrustFunctors.cuh"
#include "../../Demo/include/ParamsImageIO.hpp"

#include <thrust/device_vector.h>

using namespace ced::gpu;
static void convertToGrayScale(benchmark::State& state)
{
    ced::gpu::Image img(filename);
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        img.convertToGrayscale();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());

    }
}
BENCHMARK(convertToGrayScale)->UseManualTime();
//  ------------------------------------------------------------------------------------------------ 
static void createGaussianFilter(benchmark::State& state)
{
    float _sigma = 1.4f;
    int _dimension = 5;
    const float sigma2 = _sigma * _sigma;
    thrust::device_vector<float> filter(_dimension * _dimension);
    thrust::device_vector<int> itr(_dimension * _dimension);
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        thrust::sequence(itr.begin(), itr.end());
        thrust::transform(itr.begin(), itr.end(), filter.begin(), g(_dimension, sigma2)); 
        float sum = thrust::reduce(filter.begin(), filter.end());
        thrust::transform(filter.begin(), filter.end(), filter.begin(), multiplyConst(sum));
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());

    }
}
BENCHMARK(createGaussianFilter)->UseManualTime();
//  ------------------------------------------------------------------------------------------------ 
static void applyFilter_Kernel(benchmark::State& state)
{
    ced::gpu::Image img(filename);
    int m_width = img.getWidth();
    int m_height = img.getHeight();
    std::vector<float> m_red = img.getRedChannel();
    std::vector<float> m_green = img.getGreenChannel();
    std::vector<float> m_blue = img.getBlueChannel();
    img.convertToGrayscale();
    std::vector<float> _filter = ced::gpu::gaussianFilter(5, 1.4f);
    int _dimension = 5;
    int nwidth = m_width - _dimension  + 1;
    int nheight = m_height - _dimension + 1;
    for(auto _ : state)
    {
        thrust::device_vector<float> d_ored     = m_red;
        thrust::device_vector<float> d_ogreen   = m_green;
        thrust::device_vector<float> d_oblue    = m_blue;

        thrust::device_vector<float> d_nred(nheight * nwidth);
        thrust::device_vector<float> d_ngreen(nheight * nwidth);
        thrust::device_vector<float> d_nblue(nheight * nwidth);

        thrust::device_vector<float> d_filter = _filter;            
        // --------------------typecast raw ptr-----------------------------------
        float* d_ored_ptr   = thrust::raw_pointer_cast(d_ored.data());
        float* d_ogreen_ptr = thrust::raw_pointer_cast(d_ogreen.data());
        float* d_oblue_ptr  = thrust::raw_pointer_cast(d_oblue.data());

        float* d_nred_ptr    = thrust::raw_pointer_cast(d_nred.data());
        float* d_ngreen_ptr  = thrust::raw_pointer_cast(d_ngreen.data());
        float* d_nblue_ptr   = thrust::raw_pointer_cast(d_nblue.data());

        float* d_filter_ptr = thrust::raw_pointer_cast(d_filter.data());
        // --------------------execution config-----------------------------------
        int blockW = 32;
        int blockH = 32;
        const dim3 grid(iDivUp(nwidth, blockW),
                        iDivUp(nheight, blockH));
        const dim3 threadBlock(blockW, blockH);


        auto start = std::chrono::high_resolution_clock::now();
        d_applyFilter<<<grid, threadBlock>>>(d_ored_ptr, 
                                             d_ogreen_ptr,
                                             d_oblue_ptr,
                                             d_nred_ptr,   
                                             d_ngreen_ptr,   
                                             d_nblue_ptr,   
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
BENCHMARK(applyFilter_Kernel)->UseManualTime();
//  ------------------------------------------------------------------------------------------------ 
