#include "gtest/gtest.h"

#include "Point.hpp"

// need to return the circumcenter and the radius
TEST(CircleProperty, OutsideCircle)
{
    using namespace ced::cpu;
    // clockwise
    Point A(0,0);
    Point B(1,1);
    Point C(2,0);

    Point D(3,1);
    
    int m_00 = A.x - D.x;
    int m_01 = B.x - D.x;
    int m_02 = C.x - D.x;

    int m_10 = A.y - D.y;
    int m_11 = B.y - D.y;
    int m_12 = C.y - D.y;

    int m_20 = m_00 * m_00 + m_10 * m_10;
    int m_21 = m_01 * m_01 + m_11 * m_11;
    int m_22 = m_02 * m_02 + m_12 * m_12;

    int determinant = (m_00 * m_11 * m_22) + (m_10 * m_21 * m_02) + (m_20 * m_01 * m_12) -
                        (m_20 * m_11 * m_02) - (m_10 * m_01 * m_22) - (m_00 * m_21 * m_12); 

    // counterclockwise order -> determinant > 0 D lies inside
    // clockwise order -> determinant < 0 lies inside the circle
    // EXPECT_EQ(determinant, 8);
    EXPECT_FALSE(determinant < 0);
}

TEST(CircleProperty, insideCircle)
{
    using namespace ced::cpu;
    // clockwise
    Point A(0,0);
    Point B(1,1);
    Point C(2,0);

    Point D(1,0);
    
    int m_00 = A.x - D.x;
    int m_01 = B.x - D.x;
    int m_02 = C.x - D.x;

    int m_10 = A.y - D.y;
    int m_11 = B.y - D.y;
    int m_12 = C.y - D.y;

    int m_20 = m_00 * m_00 + m_10 * m_10;
    int m_21 = m_01 * m_01 + m_11 * m_11;
    int m_22 = m_02 * m_02 + m_12 * m_12;

    int determinant = (m_00 * m_11 * m_22) + (m_10 * m_21 * m_02) + (m_20 * m_01 * m_12) -
                        (m_20 * m_11 * m_02) - (m_10 * m_01 * m_22) - (m_00 * m_21 * m_12); 

    // counterclockwise order -> determinant > 0 D lies inside
    // clockwise order -> determinant < 0 lies inside the circle
    EXPECT_TRUE(determinant < 0);
    EXPECT_EQ(determinant, -2);
}
#include "CircumCircle.hpp"
TEST(CircleProperty, CircumRadius)
{
    using namespace ced::cpu;
    Point A(1,1);
    Point B(0,0);
    Point C(2,0);
    float ax = static_cast<float>(B.x - A.x);
    float ay = static_cast<float>(B.y - A.y);
    ASSERT_EQ(ax, -1);
    ASSERT_EQ(ay, -1);

    float bx = static_cast<float>(C.x - B.x);
    float by = static_cast<float>(C.y - B.y);
    ASSERT_EQ(bx, 2);
    ASSERT_EQ(by, 0);

    float cx = static_cast<float>(A.x - C.x);
    float cy = static_cast<float>(A.y - C.y);
    ASSERT_EQ(cx, -1);
    ASSERT_EQ(cy, 1);
    
    float axx = ax * ax;
    float ayy = ay * ay;
    ASSERT_EQ(axx, 1);
    ASSERT_EQ(ayy, 1);

    float bxx = bx * bx;
    float byy = by * by;
    ASSERT_EQ(bxx, 4);
    ASSERT_EQ(byy, 0);

    float cxx = cx * cx;
    float cyy = cy * cy;
    ASSERT_EQ(cxx, 1);
    ASSERT_EQ(cyy, 1);

    float a = std::sqrt(axx + ayy);
    ASSERT_FLOAT_EQ(a, std::sqrt(2));

    float b = std::sqrt(bxx + byy);
    ASSERT_FLOAT_EQ(b, 2.0f);

    float c = std::sqrt(cxx + cyy);
    ASSERT_FLOAT_EQ(c, std::sqrt(2));

    // T ee N 
    float T = a * b * c; 
    ASSERT_FLOAT_EQ(T, 4.0f);

    float N = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c);
    ASSERT_FLOAT_EQ(N, 16);

    float r = circumRadius(A, B, C); 
    EXPECT_FLOAT_EQ(r, 1.0f);
}

TEST(CircleProperty, CircumCenter)
{
    using namespace ced::cpu; 
    Point A(2,0);
    Point B(1,1);
    Point C(0,1);

    // looking for midpoint   
    Point midAB = {(A+B)/2}; 
    ASSERT_EQ(midAB, Point(1.5f, 0.5f));

    Point midAC = {(A+C)/2}; 
    ASSERT_EQ(midAC, Point(1.0f, 0.5f));
    
    // determine slope
    // get the negative reciprocal of the slope to get slope of the perpendicular bisector
    float slopeAB = (B.y - A.y) / (B.x - A.x); 
    slopeAB = -(1/slopeAB);
    ASSERT_FLOAT_EQ(slopeAB, 1.0f);

    float slopeAC = (C.y - A.y) / (C.x - A.x); 
    slopeAC = -(1/slopeAC);
    ASSERT_FLOAT_EQ(slopeAC, 2.0f);
    // solving mx + b = y 
    // solve for b 
    float bAB = midAB.y - slopeAB * midAB.x;
    ASSERT_FLOAT_EQ(bAB, -1.0f); 
    float bAC = midAC.y - slopeAC * midAC.x;
    ASSERT_FLOAT_EQ(bAC, -1.5f); 
    
    // solve for x 
    float x = (bAB - bAC) / (slopeAC - slopeAB);
    float y = (slopeAB * x) + bAB; 
    EXPECT_EQ(x, 0.5f);
    EXPECT_EQ(y, -0.5f);

    Point cc = circumCenter(A, B, C);
    EXPECT_EQ(cc.x, 0.5f);    
    EXPECT_EQ(cc.y, -0.5f);    
    
}
