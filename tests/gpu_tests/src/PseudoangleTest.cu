#include "gtest/gtest.h"
#include "Pseudoangle.cuh"

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace ced::gpu;

TEST(PseudoAngle, pseudoangle)
{
    Point p = {1,1};
    Point p1 = {1,2};
    Point p2 = {3,4};
    Point p3 = {5,6};
    std::vector<Point> vec = {p, p1, p2, p3};
    thrust::device_vector<Point> d_vec = vec;
    thrust::device_vector<float> d_angles(vec.size());
    std::vector<float> h_angles(vec.size());
    thrust::transform(d_vec.begin(), d_vec.end(), d_angles.begin(), angle_funct());
    thrust::copy(d_angles.begin(), d_angles.end(), h_angles.begin());
    EXPECT_EQ(h_angles[0], (3.0f - (1.0f/2.0f))/4.0f);
    EXPECT_EQ(h_angles[1], (3.0f - (1.0f/3.0f))/4.0f);
    EXPECT_EQ(h_angles[2], (3.0f - (3.0f/7.0f))/4.0f);
    EXPECT_EQ(h_angles[3], (3.0f - (5.0f/11.0f))/4.0f);
}
