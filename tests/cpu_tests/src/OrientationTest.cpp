#include "gtest/gtest.h"
#include "TriOrientation.hpp"

TEST(Orientation, CCWfunction)
{
    using namespace ced::cpu;
    Point A(0,0);
    Point B(1,1);
    Point C(2,0);
    float CW =  isCCW<float>(A,B,C);
    
    EXPECT_FALSE((CW<0)); 
}

