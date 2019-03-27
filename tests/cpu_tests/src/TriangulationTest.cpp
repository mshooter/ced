#include "gtest/gtest.h"

#include "Triangulation.hpp"
#include "Distance2P.hpp"
#include "CircumCircle.hpp"

TEST(Triangulation, chooseSeed)
{
   ced::cpu::Point p1 = {0,0};
   ced::cpu::Point p2 = {1,1};
   ced::cpu::Point p3 = {2,0};
   
   float cx = 1;
   float cy = 0.5f;

   std::vector<ced::cpu::Point> verts = {p1, p2, p3};

   unsigned int i0;
   float minDistance = std::numeric_limits<float>::max();
   unsigned int i = 0;
   for(auto point : verts)
   {
       float deltaX = cx - point.x;
       float deltaY = cy - point.y;
       float distance = std::sqrt((deltaX * deltaX) + (deltaY*deltaY));
       if(distance < minDistance)
       {
           i0 = i;
           minDistance = distance;
       }
       ++i;
   } 
   EXPECT_EQ(i0, (unsigned int)1);
 
}
//  -------------------------------------------------------------------------
TEST(Triangulation, findPointCloseToSeed)
{
    ced::cpu::Point p1 = {0,0};
    ced::cpu::Point p2 = {1,1};
    ced::cpu::Point p3 = {2,0};
    
    std::vector<ced::cpu::Point> verts = {p1, p2, p3};
    
    unsigned int i0 = 1;
    unsigned int i1;
    ASSERT_FLOAT_EQ(verts[i0].x, 1.0f); 
    ASSERT_FLOAT_EQ(verts[i0].y, 1.0f); 

    float minDistance = std::numeric_limits<float>::max();
    unsigned int i=0;
    for(auto point : verts)
    {
        if(i != i0)
        {
            float distance = ced::cpu::distance2P<float, ced::cpu::Point>(verts[i0], point);
            if(distance < minDistance && distance > 0.0f)
            {
                i1 = i; 
                minDistance = distance;
            }
        }

        ++i;
    }

    EXPECT_EQ(i1, (unsigned int)0);
    
}
//  -------------------------------------------------------------------------
//#include <iostream>
TEST(Triangulation, findThirdPointCreateCC)
{
    using namespace ced::cpu;
    Point p1 = {0,0};
    Point p2 = {1,1};
    Point p3 = {2,0};
    
    std::vector<Point> verts = {p1, p2, p3};
    unsigned int i0 = 1; 
    unsigned int i1 = 0; 
    unsigned int i2;
    
    float min_radius = std::numeric_limits<double>::max();
    float r;

    unsigned int i = 0;
    for(auto point : verts)
    {       
        // if i is not equal to 0 or 1 
        if(i != i0 && i != i1)
        { 
            //std::cout<<"POINT"<<point.x<<point.y<<std::endl;
            r = circumRadius(verts[i0], verts[i1], point); 
            if(r < min_radius )
            {
                i2 = i;
                min_radius = r;
            }
        } 
        ++i;
    }
    EXPECT_EQ(i2, (unsigned int)2);
    EXPECT_FLOAT_EQ(r, 1.0f);
}
