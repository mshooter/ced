#include "gtest/gtest.h"
#include "DrawLine.hpp"

TEST(DrawLine, calculationLine)
{
    // supposed to be a straight line
    ced::cpu::Point p1(0,0);
    ced::cpu::Point p2(1,0);
    int x_0 = p1.getX();
    int y_0 = p1.getY();

    int x_1 = p2.getX();
    int y_1 = p2.getY();

    int x = p1.getX();
    int y = p1.getY();

    int deltaX = x_1 - x_0;
    int deltaY = y_1 - y_0;

    ASSERT_EQ(deltaX, 1);
    ASSERT_EQ(deltaY, 0);

    int D = 2 * deltaY - deltaX;
    ASSERT_EQ(D, -1);

    while(x<x_1)
    {
        if(D>=0)
        {
            // put pixel -> access the container -> [(y + x * width ) * 3 + 0] = 0 
            // put pixel -> access the container -> [(y + x * width ) * 3 + 1] = 0 
            // put pixel -> access the container -> [(y + x * width ) * 3 + 2] = 1
            
            y++;
            D+=(2*deltaY-2*deltaX); 
        }
        else
        {
            // put pixel -> access the container -> [(y + x * width ) * 3 + 0] = 0 
            // put pixel -> access the container -> [(y + x * width ) * 3 + 1] = 0 
            // put pixel -> access the container -> [(y + x * width ) * 3 + 2] = 1
            D+= 2* deltaY; 
        }
        x++;
    }
    ASSERT_EQ(x, 1);
    EXPECT_EQ(D, -1);
}

TEST(DrawLine, function)
{
    std::vector<float> img(3*3*3, 0.0f);  
    int imgSize = img.size();
    ASSERT_EQ(imgSize, 27);
    ced::cpu::Point p1(0,0);
    ced::cpu::Point p2(3,3);

    ced::cpu::drawLine(p1, p2, img, 3);
    
    EXPECT_EQ(img[0], 1); // first pixel red channel
    EXPECT_EQ(img[12], 1); // diagonal pixel red channel
    EXPECT_EQ(img[24], 1); // last pixel red channel
}

