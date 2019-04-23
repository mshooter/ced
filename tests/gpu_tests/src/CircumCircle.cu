#include "gtest/gtest.h"
#include "CircumCircle.cuh"

using namespace ced::gpu;
TEST(CircumCircle, center)
{
    float2 a = make_float2(1.0f, 1.0f);
    float2 b = make_float2(2.0f, 0.0f);
    float2 c = make_float2(0.0f, 0.0f);
    
    float2 cc = circumCenter(a,b,c);
    EXPECT_FLOAT_EQ(cc.x, 0.0f);
    EXPECT_FLOAT_EQ(cc.y, 0.0f);
}
