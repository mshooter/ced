#include "gtest/gtest.h"
#include "IsPixInTri.cuh"
TEST(IsPixInTri, isPixInTri)
{
    float2 a = make_float2(0.0f, 0.0f);
    float2 b = make_float2(2.0f, 2.0f);
    float2 c = make_float2(3.0f, 1.0f);
    float2 p = make_float2(2.0f, 1.0f);
    bool result = ced::gpu::isPixInTri(a,b,c,p);
    EXPECT_TRUE(result);
}
