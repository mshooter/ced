#include "gtest/gtest.h"

#include "CCW.cuh"

using namespace ced::gpu;
TEST(isCCW, isCcw)
{
    float2 a = make_float2(0.0f, 0.0f); 
    float2 b = make_float2(1.0f, 1.0f); 
    float2 c = make_float2(2.0f, 0.0f); 
    float ccw = CCW(a,b,c);
    EXPECT_FALSE(ccw < 0);
}

