#include "gtest/gtest.h"
#include "TestKernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>

struct gaussian
{
    const int dim;
    const float sig;
    const unsigned int mid;
    gaussian(int _dim, float _sig, unsigned int _mid):dim(_dim),sig(_sig),mid(_mid){};
    __host__ __device__
         float operator()(const float& id) const { 
            int x = static_cast<int>(id)%(dim);
            int y = id / dim;
            //return x + y;
            return std::exp(-((x*x + y*y)/(2.0f*sig))) / (2.0f * std::acos(-1.0f) *sig);
        }
};

struct divideConstant
{
    const float a;
    divideConstant(float _a):a(_a){};
    __host__ __device__
         float operator()(const float& x) const { 
            return x / a;
        }
};

TEST(Thrust, gaussianFilter)
{
    int _dimension = 5;
    float _sigma = 1.0f;
    thrust::device_vector<float> filter(_dimension * _dimension);
    thrust::sequence(filter.begin(), filter.end());
    thrust::host_vector<float> hfilter(_dimension * _dimension);

    const unsigned int middle = _dimension / 2;
    const float sigma2 = _sigma * _sigma;
    // initialise 
    thrust::transform(
        filter.begin(), 
        filter.end(), 
        filter.begin(), 
        gaussian(_dimension, sigma2, middle));
    // sum
    float sum = thrust::reduce(
                    filter.begin(), 
                    filter.end(), 
                    (float)0.0f, 
                    thrust::plus<float>());
    // divide
    thrust::transform(
        filter.begin(), 
        filter.end(), 
        filter.begin(), 
        divideConstant(sum));

    thrust::copy(filter.begin(), filter.end(), hfilter.begin());
}
