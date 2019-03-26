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
