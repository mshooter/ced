#include "gtest/gtest.h"
#include "Point.hpp"

TEST(Point, constructor)
{
    ced::cpu::Point p(0,1);
    EXPECT_EQ(p.x, 0);
    EXPECT_EQ(p.y, 1);
}
//  ----------------------------------------------------
TEST(Point, compareNotEqual)
{
    ced::cpu::Point p0(0,1);
    ced::cpu::Point p1(2,1);
    
    EXPECT_FALSE(p0==p1);
}
//  ----------------------------------------------------
TEST(Point, compareEqual)
{
    ced::cpu::Point p0(2,1);
    ced::cpu::Point p1(2,1);
    
    EXPECT_TRUE(p0==p1);
}
//  ----------------------------------------------------
TEST(Point, minus)
{
    ced::cpu::Point p0(2,1);
    ced::cpu::Point p1(2,1);

    EXPECT_EQ(p0-p1, ced::cpu::Point(0,0));
}
//  ----------------------------------------------------
TEST(Point, multiply)
{
    ced::cpu::Point p0(2,1);
    ced::cpu::Point p1(2,1);
    EXPECT_EQ(p0 * p1, ced::cpu::Point(4,1));
    
}
//  ----------------------------------------------------
TEST(Point, add)
{
    ced::cpu::Point p0(2,1);
    ced::cpu::Point p1(2,1);
    EXPECT_EQ(p0 + p1, ced::cpu::Point(4,2));
    
}
//  ----------------------------------------------------
TEST(Point, equalPts)
{
    using namespace ced::cpu;
    Point p(2,2);
    Point p1(2,2);

    EXPECT_TRUE(equalPts(p,p1));
}
//  ----------------------------------------------------
TEST(Point, dotProduct)
{
    using namespace ced::cpu;
    Point a = {3,7};
    Point b = {10,12};
    
    int dotProductResult = a.x * b.x + a.y * b.y; 
    EXPECT_EQ(114, dotProductResult);
    EXPECT_EQ(dot<int>(a, b), 114);
}
