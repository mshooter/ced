#include "gtest/gtest.h"
#include <cmath>
TEST(Image, applyFilter)
{
    int width = 300; 
    int height = 400; 
    int dimW = std::ceil(static_cast<float>(width)/32); 
    int dimH = std::ceil(static_cast<float>(height)/32); 
    EXPECT_EQ(dimW, 10);
    EXPECT_EQ(dimH, 13);
}
