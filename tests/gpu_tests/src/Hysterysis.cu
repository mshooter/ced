#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>

struct isLower_bound
{
    const float min_value;
    isLower_bound(float _min_value) : min_value(_min_value){}
    __host__ __device__
    bool operator()(const float& t)
    {
        if(t < min_value)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct isUpper_bound
{
    const float max_value;
    isUpper_bound(float _max_value) : max_value(_max_value){}
    __host__ __device__
    bool operator()(const float& t)
    {
        if(t >= max_value)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct set_value
{
    const int value;
    set_value(int _value) : value(_value){}
    __host__ __device__
    float operator()(const float& t)
    {
        return value;
    }
};

TEST(Hysterysis, lowerbound)
{
    std::vector<float> h_red = {0.01f, 0.2f, 0.4f, 0.5f};
    std::vector<float> h_green = {2.0f, 0.2f, 0.4f, 0.5f};
    std::vector<float> h_blue = {0.01f, 0.2f, 4.0f, 0.5f};

    thrust::device_vector<float> d_red  = h_red;
    thrust::device_vector<float> d_green= h_green;
    thrust::device_vector<float> d_blue = h_blue;

    int minVal = 1.0f;
    int maxVal = 0.2f;
    
    // need to zip iterate
    // lower bound
    thrust::transform_if(d_green.begin(), d_green.end(), d_green.begin(), set_value(0.0f), isLower_bound(minVal));
    thrust::transform_if(d_red.begin(), d_red.end(), d_red.begin(), set_value(0.0f), isLower_bound(minVal));
    thrust::transform_if(d_blue.begin(), d_blue.end(), d_blue.begin(), set_value(0.0f), isLower_bound(minVal));

    EXPECT_FLOAT_EQ(d_green[0], 2.0f);
    EXPECT_FLOAT_EQ(d_green[1], 0.0f);
    EXPECT_FLOAT_EQ(d_green[2], 0.0f);
    EXPECT_FLOAT_EQ(d_green[3], 0.0f);

    EXPECT_FLOAT_EQ(d_red[0], 0.0f);
    EXPECT_FLOAT_EQ(d_red[1], 0.0f);
    EXPECT_FLOAT_EQ(d_red[2], 0.0f);
    EXPECT_FLOAT_EQ(d_red[3], 0.0f);

    EXPECT_FLOAT_EQ(d_blue[0], 0.0f);
    EXPECT_FLOAT_EQ(d_blue[1], 0.0f);
    EXPECT_FLOAT_EQ(d_blue[2], 4.0f);
    EXPECT_FLOAT_EQ(d_blue[3], 0.0f);

    d_red  = h_red;
    d_green= h_green;
    d_blue = h_blue;
    // upper bound
    thrust::transform_if(d_green.begin()+1, d_green.end()-1, d_green.begin()+1, set_value(1.0f), isUpper_bound(maxVal));
    thrust::transform_if(d_red.begin()  +1, d_red.end()  -1, d_red.begin()  +1, set_value(1.0f), isUpper_bound(maxVal));
    thrust::transform_if(d_blue.begin() +1, d_blue.end() -1, d_blue.begin() +1, set_value(1.0f), isUpper_bound(maxVal));
    EXPECT_FLOAT_EQ(d_green[0], 2.0f);
    EXPECT_FLOAT_EQ(d_green[1], 1.0f);
    EXPECT_FLOAT_EQ(d_green[2], 1.0f);
    EXPECT_FLOAT_EQ(d_green[3], 0.5f);

    EXPECT_FLOAT_EQ(d_red[0], 0.01f);
    EXPECT_FLOAT_EQ(d_red[1], 1.0f);
    EXPECT_FLOAT_EQ(d_red[2], 1.0f);
    EXPECT_FLOAT_EQ(d_red[3], 0.5f);

    EXPECT_FLOAT_EQ(d_blue[0], 0.01f);
    EXPECT_FLOAT_EQ(d_blue[1], 1.0f);
    EXPECT_FLOAT_EQ(d_blue[2], 1.0f);
    EXPECT_FLOAT_EQ(d_blue[3], 0.5f);
    
}
