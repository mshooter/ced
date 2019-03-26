#include <algorithm>

#include "gtest/gtest.h"
#include "Point.hpp"
#include "Compare.hpp"

TEST(Compare, compareX)
{
    using namespace ced::cpu;

    Point p1 = {4,1};
    Point p2 = {1,6};
    Point p3 = {3,2};
    std::vector<Point> verts = {p1, p2, p3};
    float min = (std::min_element(verts.begin(), verts.end(), compareX))->x;
    float max = (std::max_element(verts.begin(), verts.end(), compareX))->x;
    EXPECT_EQ(min, 1); 
    EXPECT_EQ(max, 4); 
}

TEST(Compare, compareY)
{
    using namespace ced::cpu;

    Point p1 = {4,1};
    Point p2 = {1,6};
    Point p3 = {3,2};
    std::vector<Point> verts = {p1, p2, p3};
    float min = (std::min_element(verts.begin(), verts.end(), compareY))->y;
    float max = (std::max_element(verts.begin(), verts.end(), compareY))->y;
    EXPECT_EQ(min, 1); 
    EXPECT_EQ(max, 6); 

}
