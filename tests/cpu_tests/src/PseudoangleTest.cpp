#include "gtest/gtest.h"
#include "Pseudoangle.hpp"

TEST(Pseudoangle, pseudo_angle)
{
    using namespace ced::cpu;    
    Point p(1, 5);  
    
    // 0.708...
    EXPECT_FLOAT_EQ((3.0f - 1.0f/6.0f) / 4.0f, pseudo_angle(p));
}
