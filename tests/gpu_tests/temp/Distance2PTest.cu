#include "gtest/gtest.h"
#include "Distance2P.cuh"
#include "Point.hpp"
#include "ThrustFunctors.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

TEST(Distance2P, distance2PGpu)
{
    float x1 = 0.5f
    float y1 = 0.1f
    float x2 = 0.8f
    float y2 = 0.3f


    float dist = ced::gpu::distance2P<float>(x1, y1, x2, y2);
    EXPECT_EQ(dist, 0.3f*0.3f + 0.2f*0.2f);
}

TEST(Distance2P, functor)
{
    float x1 = 0.5f
    float y1 = 0.1f
    float x2 = 0.8f
    float y2 = 0.3f
    Point p0 = {0.0f, 3.0f};
    Point p1 = {1.0f, 2.0f};
    Point p2 = {8.0f, 9.0f};
    
    ced::cpu::Point centroid = {1.0f, 1.0f};
    thrust::device_vector<float> v = {p0, p1, p2};
    thrust::device_vector<float> r;
    thrust::transform(v.begin(), v.end(), r.end(), DistanceCP(centroid));
    
    // result is distance
    EXPECT_EQ(r[0], 5.0f);
    EXPECT_EQ(r[1], 1.0f);
    EXPECT_EQ(r[2], 49.0f + 64.0f);
}
