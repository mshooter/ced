#include "gtest/gtest.h"
#include "AssignTriToPix.hpp"
#include <iostream>
TEST(AssignTriToPix, assignTriToPix)
{
    using namespace ced::cpu;
    int height = 10;
    int width = 10; 
    std::vector<Point> mpts = {Point(0,1), Point(5,5)};
    std::vector<unsigned int> pixTriIdx;
    assignTriToPix(height, width, mpts, pixTriIdx);

    EXPECT_EQ(pixTriIdx.size(), (unsigned int)100);
    EXPECT_EQ(pixTriIdx[0], (unsigned int)0);
    EXPECT_EQ(pixTriIdx[99], (unsigned int)1);
    for(auto p : pixTriIdx)
    {
        std::cout<<p<<std::endl;
    }
}
