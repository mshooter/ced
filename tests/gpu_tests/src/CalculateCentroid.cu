#include "gtest/gtest,h"
#include "CalculateCentroid.cuh"
#include <vector>

TEST(CalculateCentroid, calculateCentroid)
{
    std::vector<float> h_x = {0.0f, 1.0f, 2.0f};
    std::vector<float> h_y = {0.0f, 1.0f, 0.0f};
    thrust::device_vector<float> x = h_x;
    thrust::device_vector<float> y = h_y;
    // put function in here
    Point centroid = ced::gpu::calculateCentroid(x, y);
    EXPECT_EQ(Point(1.0f, 0.5f), centroid);
}
