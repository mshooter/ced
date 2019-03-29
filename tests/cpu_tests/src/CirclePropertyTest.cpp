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
#include "CircumCircle.hpp"
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
    EXPECT_TRUE(isPointInCircle(A, B, C, D)); 
}

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

    float r = circumRadius<Point, float>(A, B, C); 
    EXPECT_FLOAT_EQ(r, 1.0f);
}

TEST(CircleProperty, CircumCenter)
{
    using namespace ced::cpu; 
    Point P(2,0);
    Point Q(1,1);
    Point R(0,0);

    float a, b, c, e, f, g; 
    a = Q.y - P.y; 
    ASSERT_FLOAT_EQ(a, 1.0f);

    b = P.x - Q.x; 
    ASSERT_FLOAT_EQ(b, 1.0f);

    c = a*(P.x) + b*(P.y); 
    ASSERT_FLOAT_EQ(c, 2.0f);

    e = R.y - Q.y; 
    ASSERT_FLOAT_EQ(e, -1.0f);

    f = Q.x - R.x; 
    ASSERT_FLOAT_EQ(f, 1.0f);

    g = e*(Q.x) + f*(Q.y); 
    ASSERT_FLOAT_EQ(g, 0.0f);


    Point mp = ((P+Q)/2.0f);
    ASSERT_EQ(mp.x, 1.5f);
    ASSERT_EQ(mp.y, 0.5f);

    c = -b * (mp.x) + a * (mp.y);
    ASSERT_EQ(c, -1.0f);

    b = -b;
    std::swap(a,b);
    ASSERT_EQ(a, -1.0f);
    ASSERT_EQ(b, 1.0f);

    Point mpp = ((Q+R) / 2.0f);
    ASSERT_EQ(mpp.x, 0.5f);
    ASSERT_EQ(mpp.y, 0.5f);

    g = -f * (mpp.x) + e * (mpp.y);
    ASSERT_EQ(g, -1.0f);
    
    f = -f;
    std::swap(e,f);
    ASSERT_EQ(e, -1.0f);
    ASSERT_EQ(f, -1.0f);

    float D = a * f - e * b;
    ASSERT_EQ(D, 2);

    if(D!=0)
    {
        float x = (f*c - b*g) / D;
        ASSERT_EQ(x, 1);
        float y = (a*g - e*c) / D;
        ASSERT_EQ(y, 0);
    }

    Point cc = circumCenter<Point>(P, Q, R);
    EXPECT_EQ(cc.x, 1);    
    EXPECT_EQ(cc.y, 0);    
    
}
