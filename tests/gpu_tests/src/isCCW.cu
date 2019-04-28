#include "gtest/gtest.h"

#include "CCW.cuh"

using namespace ced::gpu;
TEST(isCCW, isCcw)
{
    float ax = 0.0f; 
    float bx = 1.0f; 
    float cx = 2.0f; 
    float ay = 0.0f;
    float by = 1.0f;
    float cy = 0.0f;
    float ccw = CCW(ax, ay, bx, by, cx, cy);
    EXPECT_FALSE(ccw < 0);
}

