#include "gtest/gtest.h"
#include "Point.hpp"
#include "IsPixInTri.hpp"
// https://observablehq.com/@kelleyvanevert/2d-point-in-triangle-test
TEST(CheckIfPointInTri, checkIfPointInTri)
{
    using namespace ced::cpu;
    Point a = {0,0};
    Point b = {1,2};
    Point c = {4,0};

    Point p = {1,1};
    // compute vectors 
    Point v0 = c - a;      
    Point v1 = b - a;      
    Point v2 = p - a;      
    ASSERT_EQ(v0, Point(4,0));
    ASSERT_EQ(v1, Point(1,2));
    ASSERT_EQ(v2, Point(1,1));
    // compute dot product
    int dot00 = dot<int>(v0, v0);
    int dot01 = dot<int>(v1, v0);
    int dot02 = dot<int>(v0, v2);
    int dot11 = dot<int>(v1, v1);
    int dot12 = dot<int>(v1, v2);
    ASSERT_EQ(dot00, 16);
    ASSERT_EQ(dot01, 4);
    ASSERT_EQ(dot02, 4);
    ASSERT_EQ(dot11, 5);
    ASSERT_EQ(dot12, 3);
    // compute barycentric coordinates
    float invDenom = 1.0f/static_cast<float>(dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    ASSERT_FLOAT_EQ(invDenom, 1.0f/64.0f);
    ASSERT_FLOAT_EQ(u, 8.0f*invDenom);
    ASSERT_FLOAT_EQ(v, 32.0f*invDenom);
    // check if point is in triangle
    EXPECT_TRUE((u>=0)&&(v>=0)&&(u+v<1));
    EXPECT_TRUE(isPixInTri(a,b,c,p));
}

TEST(CheckIfPointInTri, checkPointInTriv2)
{
    using namespace ced::cpu;
    Point a = {4,4};
    Point b = {0,4};
    Point c = {0,0};

    Point p = {0,0};
    // compute vectors 
    Point v0 = c - a;      
    Point v1 = b - a;      
    Point v2 = p - a;      
    ASSERT_EQ(v0, Point(-4,-4));
    ASSERT_EQ(v1, Point(-4,0));
    ASSERT_EQ(v2, Point(-4,-4));
    // compute dot product
    int dot00 = dot<int>(v0, v0);
    int dot01 = dot<int>(v1, v0);
    int dot02 = dot<int>(v0, v2);
    int dot11 = dot<int>(v1, v1);
    int dot12 = dot<int>(v1, v2);
    ASSERT_EQ(dot00, 32);
    ASSERT_EQ(dot01, 16);
    ASSERT_EQ(dot02, 32);
    ASSERT_EQ(dot11, 16);
    ASSERT_EQ(dot12, 16);
    // compute barycentric coordinates
    float invDenom = 1.0f/static_cast<float>(dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    ASSERT_FLOAT_EQ(invDenom, 1.0f/256.0f);
    ASSERT_FLOAT_EQ(u, 256.0f*invDenom);
    ASSERT_FLOAT_EQ(v, 0.0f*invDenom);
    // check if point is in triangle
    EXPECT_TRUE((u>=0)&&(v>=0)&&(u+v<=1));
    EXPECT_TRUE(isPixInTri(a,b,c,p));
}

