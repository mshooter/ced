#include "gtest/gtest.h"
#include "CircumCircle.cuh"

using namespace ced::gpu;
TEST(CircumCircle, center)
{
    float ax = 0.0f; 
    float bx = 1.0f; 
    float cx = 2.0f; 
    float ay = 0.0f;
    float by = 1.0f;
    float cy = 0.0f;
    
    float x = 0.0f;
    float y = 0.0f;
    circumCenter(ax, ay, bx, by, cx, cy, x, y);
    EXPECT_FLOAT_EQ(x, 1.0f);
    EXPECT_FLOAT_EQ(y, -1.0f);
}
