#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <vector>

struct calculate_intensity
{
    __host__ __device__
    float operator()(const thrust::tuple<float, float, float>& t)
    {
        return (thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t))/3.0f;
    }
};
TEST(Gradient, getItensity)
{
    std::vector<float> h_red = {0.5f, 0.6f, 0.2f}; 
    std::vector<float> h_green = {0.1f, 0.2f, 0.3f}; 
    std::vector<float> h_blue = {0.0f, 0.1f, 0.4f}; 
    thrust::device_vector<float> d_red = h_red;
    thrust::device_vector<float> d_green = h_green;
    thrust::device_vector<float> d_blue = h_blue;
    // amount of pixels
    thrust::device_vector<float> d_intensity(d_red.size());
    thrust::transform(
                        thrust::make_zip_iterator(
                                                    thrust::make_tuple(d_red.begin(), d_green.begin(), d_blue.begin())),
                        thrust::make_zip_iterator( 
                                                    thrust::make_tuple(d_red.end(), d_green.end(), d_blue.end())),
                        d_intensity.begin(),
                        calculate_intensity());
    
    EXPECT_FLOAT_EQ(d_intensity[0], 0.2f);
    EXPECT_FLOAT_EQ(d_intensity[1], 0.3f);
    EXPECT_FLOAT_EQ(d_intensity[2], 0.3f);
}
//  -------------------------------------------------------------------------------------
struct calculate_magnitude
{
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t)
    {
        return fabs(thrust::get<0>(t)) + fabs(thrust::get<1>(t));
    }
};
TEST(Gradient, calculateMagnitude)
{
    std::vector<float> h_xgradient = {0.1f, 0.2f, 0.3f};
    std::vector<float> h_ygradient = {0.2f, 0.3f, 0.4f};
    int amountOfPixels = 3;
    
    thrust::device_vector<float> d_xgradient = h_xgradient;
    thrust::device_vector<float> d_ygradient = h_ygradient;
    
    thrust::device_vector<float> d_magnitude(amountOfPixels);
    thrust::transform(
                        thrust::make_zip_iterator(
                                                    thrust::make_tuple(d_xgradient.begin(), d_ygradient.begin())),
                        thrust::make_zip_iterator( 
                                                    thrust::make_tuple(d_xgradient.end(), d_ygradient.end())),
                        d_magnitude.begin(),
                        calculate_magnitude());

    EXPECT_FLOAT_EQ(d_magnitude[0], 0.3f);
    EXPECT_FLOAT_EQ(d_magnitude[1], 0.5f);
    EXPECT_FLOAT_EQ(d_magnitude[2], 0.7f);
}
//  -------------------------------------------------------------------------------------
struct calculate_orientation
{
    // calculate the gradient direction in terms of degrees and make it positive 
    __host__ __device__
    int operator()(const thrust::tuple<float, float>& t)
    {
        return fabs(atan(thrust::get<1>(t)/thrust::get<0>(t)) * 180.0f/ 3.142f);
    }
};
TEST(Gradient, calculateOrientation)
{   
    std::vector<float> h_xgradient = {0.1f, 0.2f, 0.3f};
    std::vector<float> h_ygradient = {0.2f, 0.3f, 0.4f};
    int amountOfPixels = 3;
    thrust::device_vector<float> d_xgradient = h_xgradient;
    thrust::device_vector<float> d_ygradient = h_ygradient;
    
    thrust::device_vector<float> d_orientation(amountOfPixels);    
    thrust::transform(
                        thrust::make_zip_iterator(
                                                    thrust::make_tuple(d_xgradient.begin(), d_ygradient.begin())),
                        thrust::make_zip_iterator( 
                                                    thrust::make_tuple(d_xgradient.end(), d_ygradient.end())),
                        d_orientation.begin(),
                        calculate_orientation());
    EXPECT_EQ(d_orientation[0],  63);
    EXPECT_EQ(d_orientation[1],  56);
    EXPECT_EQ(d_orientation[2],  53);
}
//  -------------------------------------------------------------------------------------

