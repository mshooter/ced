#include "gtest/gtest.h"
#include "Point.hpp"
#include "TriOrientation.hpp"
TEST(Orientation, CCW)
{
    using namespace ced::cpu; 
    Point C(0,0);
    Point B(1,1);
    Point A(2,0);
    
    float yEdgeRho = static_cast<float>(B.y - A.y); 
    float xEdgeRho = static_cast<float>(B.x - A.x); 

    float rho = yEdgeRho / xEdgeRho;
    ASSERT_FLOAT_EQ(rho, -1.0f);

    float yEdgeTau = static_cast<float>(C.y - B.y); 
    float xEdgeTau = static_cast<float>(C.x - B.x); 

    float tau = yEdgeTau / xEdgeTau;
    ASSERT_FLOAT_EQ(tau, 1.0f);

    // if it is zero it is colinear
    EXPECT_TRUE((rho-tau) < 0);
}

TEST(Orientation, CCWfunction)
{
    using namespace ced::cpu;
    Point A(0,0);
    Point B(1,1);
    Point C(2,0);
    bool tri =  isCCW(A,B,C);
    
    EXPECT_FALSE(tri); 
}
