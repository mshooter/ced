#include "gtest/gtest.h"
#include <iostream>
#include "SplitVector.hpp"
#include "Point.hpp"

TEST(SplitVector, splitVector)
{
    // vector with the information 
    std::vector<int> vec = {0, 1, 2, 3, 4, 5, 6, 7};
    int sizeVec = vec.size();
    int midIndex = sizeVec / 2;
    // need vector with the sub vectors
    std::vector<std::vector<int>> finalVec;

    std::vector<int>::const_iterator begin = vec.begin();
    std::vector<int>::const_iterator middle = vec.begin() + midIndex;
    std::vector<int>::const_iterator end = vec.begin() + sizeVec;
    std::vector<int> nVecL(begin, middle);
    std::vector<int> mVecR(middle, end);

    EXPECT_EQ(nVecL[0], 0);
    EXPECT_EQ(nVecL[1], 1);
    EXPECT_EQ(int(nVecL.size()), 4);
    EXPECT_EQ(nVecL[2], 2);
    EXPECT_EQ(nVecL[3], 3);

    EXPECT_EQ(mVecR[0], 4);
    EXPECT_EQ(mVecR[1], 5);
    EXPECT_EQ(mVecR[2], 6);
    EXPECT_EQ(mVecR[3], 7);
}

TEST(SplitVector, splitVectorFunction)
{
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int>> finalVec;
    ced::cpu::splitVector(finalVec, vec);
    int sizeFinal = finalVec.size();
    EXPECT_EQ(sizeFinal, 4);
    std::vector<int> vec_0 = finalVec[0];
    std::vector<int> vec_1 = finalVec[1];
    std::vector<int> vec_2 = finalVec[2];
    std::vector<int> vec_3 = finalVec[3];

    EXPECT_EQ(vec_0[0], 1);
    EXPECT_EQ(vec_0[1], 2);

    EXPECT_EQ(vec_1[0], 3);
    EXPECT_EQ(vec_1[1], 4);

    EXPECT_EQ(vec_2[0], 5);
    EXPECT_EQ(vec_2[1], 6);

    EXPECT_EQ(vec_3[0], 7);
    EXPECT_EQ(vec_3[1], 8);
    EXPECT_EQ(vec_3[2], 9);
}

TEST(SplitVector, splitVectorFunctionPts)
{
    ced::cpu::Point p1(0,0); 
    ced::cpu::Point p2(0,1); 

    ced::cpu::Point p3(2,0); 
    ced::cpu::Point p4(4,0); 

    ced::cpu::Point p5(1,0); 
    ced::cpu::Point p6(10,0); 

    std::vector<ced::cpu::Point> pts = {p1, p2, p3, p4, p5, p6};
    std::vector<std::vector<ced::cpu::Point>> fvec;

    ced::cpu::splitVector(fvec, pts);
    int fsize = fvec.size();
    EXPECT_EQ(fsize, 2);

    EXPECT_EQ(fvec[0][0].getX(), 0);
    EXPECT_EQ(fvec[0][0].getY(), 0);
    EXPECT_EQ(fvec[0][1].getX(), 0);
    EXPECT_EQ(fvec[0][1].getY(), 1);
    EXPECT_EQ(fvec[0][2].getX(), 2);
    
    EXPECT_EQ(fvec[1][0].getX(), 4);
    EXPECT_EQ(fvec[1][1].getX(), 1);
    EXPECT_EQ(fvec[1][2].getX(), 10);
}
