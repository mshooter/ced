#include "gtest/gtest.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "CalculateCentroid.cuh"
#include <vector>

TEST(CalculateCentroid, calculateCentroid)
{
    std::vector<float> h_x = {0.0f, 1.0f, 2.0f};
    std::vector<float> h_y = {0.0f, 1.0f, 0.0f};
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;
    ced::gpu::Point centroid = ced::gpu::calculateCentroid(d_x, d_y);
    EXPECT_EQ(centroid.x, 1.0f);    
    EXPECT_EQ(centroid.y, 0.5f);    
    float minx = d_x[thrust::min_element(thrust::device, d_x.begin(), d_x.end())- d_x.begin()];
    float maxx = d_x[thrust::max_element(thrust::device, d_x.begin(), d_x.end())- d_x.begin()];
    float miny = d_y[thrust::min_element(thrust::device, d_y.begin(), d_y.end())- d_y.begin()];
    float maxy = d_y[thrust::max_element(thrust::device, d_y.begin(), d_y.end())- d_y.begin()];
    // put function in here
    //Point centroid = ced::gpu::calculateCentroid(x, y);
    EXPECT_FLOAT_EQ(minx, 0.0f);
    EXPECT_FLOAT_EQ(maxx, 2.0f);
    EXPECT_FLOAT_EQ(miny, 0.0f);
    EXPECT_FLOAT_EQ(maxy, 1.0f);
    
    float cx = (minx + maxx) / 2.0f;
    float cy = (miny + maxy) / 2.0f;
    
    EXPECT_EQ(cx, 1.0f);    
    EXPECT_EQ(cy, 0.5f);    
}
