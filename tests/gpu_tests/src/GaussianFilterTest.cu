#include "gtest/gtest.h"
#include "GaussianFilter.cuh"
TEST(GaussianFilter, helloWorld)
{
    using namespace ced::gpu;
    helloWorld<<<1,1>>>();
    // no output otherwise
    cudaDeviceSynchronize();
}
