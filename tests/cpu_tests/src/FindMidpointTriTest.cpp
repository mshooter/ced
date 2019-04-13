#include "gtest/gtest.h"
#include "FindMidpointTri.hpp"

TEST(FindMidpointTri, getMPlist)
{
    using namespace ced::cpu;
    Point p0 = {0,0};
    Point p1 = {1,1};
    Point p2 = {2,0};
    std::vector<Point> v = {p0, p1, p2};
    std::vector<unsigned int> t = {2,1,0};
    std::vector<Point> mp;
    findMidpointTri(v, t, mp);
    EXPECT_EQ(mp.size(), (unsigned int) 1);
    EXPECT_EQ(mp[0].x, 1);
    EXPECT_EQ(mp[0].y, 0);
}
