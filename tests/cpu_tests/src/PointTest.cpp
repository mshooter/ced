#include "gtest/gtest.h"
#include "Point.hpp"

TEST(Point, getX)
{
    ced::cpu::Point p(4, 5);
    EXPECT_EQ(4, p.getX());
}
//--------------------------------------------------------------------------------
TEST(Point, getY)
{
    ced::cpu::Point p(4, 5);
    EXPECT_EQ(5, p.getY());
}
//--------------------------------------------------------------------------------
TEST(Point, setX)
{
    ced::cpu::Point p;
    p.setX(3);
    EXPECT_EQ(3, p.getX());
}
//--------------------------------------------------------------------------------
TEST(Point, setY)
{
    ced::cpu::Point p;
    p.setY(4);
    EXPECT_EQ(4, p.getY());
}
//--------------------------------------------------------------------------------


