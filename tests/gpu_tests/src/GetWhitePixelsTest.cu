#include "gtest/gtest.h"

#include <vector>

#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>

struct add_three_vectors
{
    __host__ __device__ 
    float operator()(const thrust::tuple<float, float, float>& t)
    {
        return thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t);
    }
};

struct isWhite
{
    __host__ __device__ 
    int operator()(const float& f)
    {
        if(f == 3.0f)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
};
struct isIdentity
{
    __host__ __device__ 
    int operator()(const thrust::tuple<int, int>& f)
    {
        if(thrust::get<0>(f) == 1)
        {
            return thrust::get<1>(f);
        }
        else
        {
            return -1;
        }
    }
};
struct isNegative
{
    __host__ __device__ 
    int operator()(const int& f)
    {
        return f < 0;
    }
};




TEST(GetWhitePixels, add_three_vectors)
{   
    std::vector<float> h_red = {1.0f, 0.2f, 1.0f};
    std::vector<float> h_green = {1.0f, 0.2f, 1.0f};
    std::vector<float> h_blue = {1.0f, 0.2f, 1.0f};

    thrust::device_vector<float> d_red = h_red;
    thrust::device_vector<float> d_green = h_green;
    thrust::device_vector<float> d_blue = h_blue;

    thrust::device_vector<float> d_result(3, 0.0f);

    thrust::transform(  
                        thrust::make_zip_iterator(  thrust::make_tuple(d_red.begin(), d_green.begin(), d_blue.begin())),
                        thrust::make_zip_iterator(  thrust::make_tuple(d_red.end(), d_green.end(), d_blue.end())),
                        d_result.begin(),
                        add_three_vectors());

    EXPECT_FLOAT_EQ(d_result[0], 3.0f);
    EXPECT_FLOAT_EQ(d_result[1], 0.6f);
    EXPECT_FLOAT_EQ(d_result[2], 3.0f);
   
    thrust::device_vector<int> dummy_vector(3, 0);

    thrust::transform(d_result.begin(), d_result.end(), dummy_vector.begin(), isWhite()); 
    EXPECT_EQ(dummy_vector[0], 1);
    EXPECT_EQ(dummy_vector[1], 0);
    EXPECT_EQ(dummy_vector[2], 1);
    
    thrust::device_vector<int> white_pixels(3,-1);
    thrust::device_vector<int> index(3);
    thrust::sequence(index.begin(), index.end());
    ASSERT_EQ(index[0],0);
    ASSERT_EQ(index[1],1);
    ASSERT_EQ(index[2],2);

    thrust::transform(  thrust::make_zip_iterator( thrust::make_tuple(dummy_vector.begin(), index.begin())),
                        thrust::make_zip_iterator( thrust::make_tuple(dummy_vector.end(), index.end())),
                        white_pixels.begin(),
                        isIdentity());    

    EXPECT_EQ(white_pixels[0], 0);
    EXPECT_EQ(white_pixels[1], -1);
    EXPECT_EQ(white_pixels[2], 2);
    
    std::vector<int> white_pix_host(3);
    auto itr = thrust::remove_if(white_pixels.begin(), white_pixels.end(), isNegative()); 
    int size = thrust::distance(white_pixels.begin(), itr);

    white_pix_host.resize(size);
    ASSERT_EQ(white_pix_host.size(), 2);

    thrust::copy(white_pixels.begin(), itr, white_pix_host.begin());
    EXPECT_EQ(white_pix_host[0],0);
    EXPECT_EQ(white_pix_host[1],2);
    ASSERT_EQ(white_pix_host.size(), 2);
}
    
