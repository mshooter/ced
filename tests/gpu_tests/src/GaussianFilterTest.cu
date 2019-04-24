#include "gtest/gtest.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>



struct g
{
    const int d;
    const float s; 
    g(int _d, float _s) : d(_d), s(_s) {}  
    __host__ __device__ 
    float operator()(const int& id)
    {
        int middle = d / 2;
        int y = (id / d) - middle; 
        int x = (id % d) - middle; 
        return std::exp(-((x*x + y*y)/(2.0f*s))) / (2.0f * std::acos(-1.0f) *s);
    }
};

struct multiplyConst
{
    const float a; 
    multiplyConst(float _a) : a(_a) {}
    __host__ __device__ 
    float operator()(const float& id)
    {
        return id / a;
    }
    
};

#include "GaussianFilter.cuh"
TEST(GaussianFilter, gaussianfilterThrust)
{
    int _dimension = 5;
    float _sigma = 1.4f;
    const float sigma2 = _sigma * _sigma;
    std::vector<float> gfilter = ced::gpu::gaussianFilter(_dimension, _sigma);
    EXPECT_NEAR(gfilter[0], 0.0128f, 0.01f);
    EXPECT_NEAR(gfilter[1], 0.0267f, 0.01f);
    EXPECT_NEAR(gfilter[2], 0.0341f, 0.01f);
    EXPECT_NEAR(gfilter[3], 0.0257f, 0.01f);
    EXPECT_NEAR(gfilter[4], 0.0128f, 0.01f);
 
    EXPECT_NEAR(gfilter[5], 0.0267f, 0.01f);
    EXPECT_NEAR(gfilter[6], 0.0556f, 0.01f);
    EXPECT_NEAR(gfilter[7], 0.0711f, 0.01f);
    EXPECT_NEAR(gfilter[8], 0.0556f, 0.01f);
    EXPECT_NEAR(gfilter[9], 0.0267f, 0.01f);
    
    EXPECT_NEAR(gfilter[10], 0.0341f, 0.01f);
    EXPECT_NEAR(gfilter[11], 0.0711f, 0.01f);
    EXPECT_NEAR(gfilter[12], 0.0908f, 0.01f);
    EXPECT_NEAR(gfilter[13], 0.0711f, 0.01f);
    EXPECT_NEAR(gfilter[14], 0.0341f, 0.01f);
 
    EXPECT_NEAR(gfilter[15], 0.0267f, 0.01f);
    EXPECT_NEAR(gfilter[16], 0.0556f, 0.01f);
    EXPECT_NEAR(gfilter[17], 0.0711f, 0.01f);
    EXPECT_NEAR(gfilter[18], 0.0556f, 0.01f);
    EXPECT_NEAR(gfilter[19], 0.0267f, 0.01f);
 
    EXPECT_NEAR(gfilter[20], 0.0128f, 0.01f);
    EXPECT_NEAR(gfilter[21], 0.0267f, 0.01f);
    EXPECT_NEAR(gfilter[22], 0.0341f, 0.01f);
    EXPECT_NEAR(gfilter[23], 0.0267f, 0.01f);
    EXPECT_NEAR(gfilter[24], 0.0128f, 0.01f);
    thrust::device_vector<float> filter(_dimension * _dimension);
    thrust::device_vector<int> itr(_dimension * _dimension);
    thrust::sequence(itr.begin(), itr.end());
    thrust::transform(itr.begin(), itr.end(), filter.begin(), g(_dimension, sigma2)); 

    float sum = thrust::reduce(filter.begin(), filter.end());
    thrust::device_vector<float> d_filter(_dimension * _dimension);
    thrust::transform(filter.begin(), filter.end(), filter.begin(), multiplyConst(sum));
    std::vector<float> h_filter(filter.begin(), filter.end());

    EXPECT_NEAR(h_filter[0], 0.0128f, 0.01f);
    EXPECT_NEAR(h_filter[1], 0.0267f, 0.01f);
    EXPECT_NEAR(h_filter[2], 0.0341f, 0.01f);
    EXPECT_NEAR(h_filter[3], 0.0257f, 0.01f);
    EXPECT_NEAR(h_filter[4], 0.0128f, 0.01f);
 
    EXPECT_NEAR(h_filter[5], 0.0267f, 0.01f);
    EXPECT_NEAR(h_filter[6], 0.0556f, 0.01f);
    EXPECT_NEAR(h_filter[7], 0.0711f, 0.01f);
    EXPECT_NEAR(h_filter[8], 0.0556f, 0.01f);
    EXPECT_NEAR(h_filter[9], 0.0267f, 0.01f);
    
    EXPECT_NEAR(h_filter[10], 0.0341f, 0.01f);
    EXPECT_NEAR(h_filter[11], 0.0711f, 0.01f);
    EXPECT_NEAR(h_filter[12], 0.0908f, 0.01f);
    EXPECT_NEAR(h_filter[13], 0.0711f, 0.01f);
    EXPECT_NEAR(h_filter[14], 0.0341f, 0.01f);
 
    EXPECT_NEAR(h_filter[15], 0.0267f, 0.01f);
    EXPECT_NEAR(h_filter[16], 0.0556f, 0.01f);
    EXPECT_NEAR(h_filter[17], 0.0711f, 0.01f);
    EXPECT_NEAR(h_filter[18], 0.0556f, 0.01f);
    EXPECT_NEAR(h_filter[19], 0.0267f, 0.01f);
 
    EXPECT_NEAR(h_filter[20], 0.0128f, 0.01f);
    EXPECT_NEAR(h_filter[21], 0.0267f, 0.01f);
    EXPECT_NEAR(h_filter[22], 0.0341f, 0.01f);
    EXPECT_NEAR(h_filter[23], 0.0267f, 0.01f);
    EXPECT_NEAR(h_filter[24], 0.0128f, 0.01f);
}
