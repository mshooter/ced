#include "benchmark/benchmark.h"
#include "Pseudoangle.cuh"

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <chrono>

using namespace ced::gpu;
static void BM_Pseudoangle(benchmark::State& state)
{

    Point p = {1,1};
    Point p1 = {1,2};
    Point p2 = {3,4};
    Point p3 = {5,6};
    std::vector<Point> vec = {p, p1, p2, p3};
    thrust::device_vector<Point> d_vec = vec;
    thrust::device_vector<float> d_angles(vec.size());
    std::vector<float> h_angles(vec.size());
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        thrust::transform(d_vec.begin(), d_vec.end(), d_angles.begin(), angle_funct());
        //thrust::copy(d_angles.begin(), d_angles.end(), h_angles.begin());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_Pseudoangle)->UseManualTime();
