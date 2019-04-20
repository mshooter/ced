#include "gtest/gtest.h"
#include "Point.hpp"

TEST(CalcTriMidpoint, calculateMidpoint)
{
    using namespace ced::cpu;
    Point a = {0,0};    
    Point b = {1,1};    
    Point c = {2,0};    
    
    float x = (a.x + b.x + c.x)/3.0f;
    float y = (a.y + b.y + c.y)/3.0f;

    Point res = {1.0f, 1.0f/3.0f};

    EXPECT_EQ(res, Point(x,y));
}
