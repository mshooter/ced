#include "gtest/gtest.h"

struct p
{
    int x;
    int y;
};

TEST(EmptyCircleProperty, OutsideCircle)
{
    // clockwise
    p A = {0,0};
    p B = {1,1};
    p C = {2,0};

    p D = {3,1};
    
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
    EXPECT_EQ(determinant, 8);
}

TEST(EmptyCircleProperty, insideCircle)
{
    // clockwise
    p A = {0,0};
    p B = {1,1};
    p C = {2,0};

    p D = {1,0};
    
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
    EXPECT_EQ(determinant, -2);
}

