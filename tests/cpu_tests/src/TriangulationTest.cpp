#include "gtest/gtest.h"

#include "Triangulation.hpp"
#include <cmath>

 TEST(Triangulation, minimumDistanceCentroid)
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
    
