#include "gtest/gtest.h"
#include "Point.cuh"

using namespace ced::gpu;

TEST(Point, constructor)
{
    Point p = {0.2f, 0.4f};    
    EXPECT_EQ(p.x, 0.2f);
    EXPECT_EQ(p.y, 0.4f);
}
// ------------------------------------------------------------
TEST(Point, compareNotEqual)
{
    Point p0(0,1);
    Point p1(2,1);

    EXPECT_FALSE(p0==p1);
}
// ------------------------------------------------------------
TEST(Point, compareEqual)
{
    Point p0(4,1);
    Point p1(4,1);

    EXPECT_TRUE(p0==p1);
}
// ------------------------------------------------------------
TEST(Point, minus)
{
    Point p0(2,1);
    Point p1(2,1);

    EXPECT_EQ(p0-p1, Point(0,0));
}
//  ----------------------------------------------------
TEST(Point, multiply)
{
    Point p0(2,1);
    Point p1(2,1);
    EXPECT_EQ(p0 * p1, Point(4,1));
    
}
//  ----------------------------------------------------
TEST(Point, add)
{
    Point p0(2,1);
    Point p1(2,1);
    EXPECT_EQ(p0 + p1, Point(4,2));
    
}
//  ----------------------------------------------------
TEST(Point, equalPts)
{
    Point p(2,2);
    Point p1(2,2);

    EXPECT_TRUE(equalPts(p,p1));
}
//  ----------------------------------------------------
TEST(Point, dotProduct)
{
    Point a = {3,7};
    Point b = {10,12};
    
    int dotProductResult = a.x * b.x + a.y * b.y; 
    EXPECT_EQ(114, dotProductResult);
    EXPECT_EQ(dot<int>(a, b), 114);
}

