#include "gtest/gtest.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <vector>
#include <iostream>

#include <cuda.h>
#include "ConvertToGrayScale.cuh"

struct divideByConstant
{
    const float a;
    divideByConstant(float _a) : a(_a) {}
    __device__ 
    float operator()(const float& element)
    {
        return element/a;
    }
};

struct addThreeVects
{
    template <typename T>
    __device__ 
    void operator()(T t)
    {
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t);
    }
};



TEST(Grayscale, grayscale)
{   
    // host
    std::vector<float> r = {0.1f, 0.4f, 0.7f};
    std::vector<float> g = {0.2f, 0.5f, 0.8f};
    std::vector<float> b = {0.3f, 0.6f, 0.9f};

    // device
    thrust::device_vector<float> red(r); 
    thrust::device_vector<float> green(g); 
    thrust::device_vector<float> blue(b); 
    thrust::device_vector<float> result(3);
    thrust::fill(result.begin(), result.end(), 0);
    
    ced::gpu::convertToGrayScale(red, green, blue);
    std::vector<float> hred(red.begin(), red.end()); 
    std::vector<float> hgreen(green.begin(), green.end()); 
    std::vector<float> hblue(blue.begin(), blue.end()); 

    EXPECT_FLOAT_EQ(hred[0], 0.6f/3.0f);
    EXPECT_FLOAT_EQ(hred[1], 1.5f/3.0f);
    EXPECT_FLOAT_EQ(hred[2], 2.4f/3.0f);

    EXPECT_FLOAT_EQ(hgreen[0], 0.6f/3.0f);
    EXPECT_FLOAT_EQ(hgreen[1], 1.5f/3.0f);
    EXPECT_FLOAT_EQ(hgreen[2], 2.4f/3.0f);

    EXPECT_FLOAT_EQ(hblue[0], 0.6f/3.0f);
    EXPECT_FLOAT_EQ(hblue[1], 1.5f/3.0f);
    EXPECT_FLOAT_EQ(hblue[2], 2.4f/3.0f);
    
 //  // sum red and green
 //  thrust::for_each(
 //      thrust::make_zip_iterator(thrust::make_tuple(red.begin(), green.begin(), blue.begin(), result.begin())),
 //      thrust::make_zip_iterator(thrust::make_tuple(red.end(), green.end(), blue.end(), result.end())),
 //      addThreeVects());
 //  std::vector<float> hresult(result.begin(), result.end()); 
 //  ASSERT_FLOAT_EQ(hresult[0], 0.6f);
 //  ASSERT_FLOAT_EQ(hresult[1], 1.5f);
 //  ASSERT_FLOAT_EQ(hresult[2], 2.4f);
 //  // DIVIDE
 //  thrust::transform(result.begin(), result.end(), result.begin(), divideByConstant(3.0f));
 //  hresult = std::vector<float>(result.begin(), result.end()); 
 //  EXPECT_FLOAT_EQ(hresult[0], 0.6f/3.0f);
 //  EXPECT_FLOAT_EQ(hresult[1], 1.5f/3.0f);
 //  EXPECT_FLOAT_EQ(hresult[2], 2.4f/3.0f);

 //  // assign result element to red, green and blue
 //  thrust::copy(result.begin(), result.end(), red.begin());    
 //  thrust::copy(result.begin(), result.end(), green.begin());    
 //  thrust::copy(result.begin(), result.end(), blue.begin());    

 //  std::vector<float> hred(red.begin(), red.end()); 
 //  std::vector<float> hgreen(green.begin(), green.end()); 
 //  std::vector<float> hblue(blue.begin(), blue.end()); 

 //  EXPECT_FLOAT_EQ(hred[0], 0.6f/3.0f);
 //  EXPECT_FLOAT_EQ(hred[1], 1.5f/3.0f);
 //  EXPECT_FLOAT_EQ(hred[2], 2.4f/3.0f);

 //  EXPECT_FLOAT_EQ(hgreen[0], 0.6f/3.0f);
 //  EXPECT_FLOAT_EQ(hgreen[1], 1.5f/3.0f);
 //  EXPECT_FLOAT_EQ(hgreen[2], 2.4f/3.0f);

 //  EXPECT_FLOAT_EQ(hblue[0], 0.6f/3.0f);
 //  EXPECT_FLOAT_EQ(hblue[1], 1.5f/3.0f);
 //  EXPECT_FLOAT_EQ(hblue[2], 2.4f/3.0f);
}
