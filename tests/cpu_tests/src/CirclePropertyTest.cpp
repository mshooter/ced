#include "gtest/gtest.h"

#include "CircumCircle.hpp"

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
    EXPECT_TRUE(isPointInCircle<float>(A, B, C, D)); 
}

TEST(CircleProperty, CircumRadius)
{
    using namespace ced::cpu;
    Point A(1,1);
    Point B(0,0);
    Point C(2,0);

    float r = circumRadius<float>(A, B, C); 
    EXPECT_FLOAT_EQ(r, 0.0f);
}

TEST(CircleProperty, CircumCenter)
{
    using namespace ced::cpu; 
    Point P(1,1);
    Point Q(0,0);
    Point R(2,0);

    Point cc = circumCenter(P, Q, R);
    EXPECT_EQ(cc.x, 1);    
    EXPECT_EQ(cc.y, 1);    
}
