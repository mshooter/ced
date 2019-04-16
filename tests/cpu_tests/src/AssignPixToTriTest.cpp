#include "gtest/gtest.h"
#include "Point.hpp"
#include "IsPixInTri.hpp"
#include "AssignPixToTri.hpp"
#include <iostream>
#include <map>
#include <algorithm>
#include <iterator>

TEST(AssignPixToTri, assignPixToTri)
{
    using namespace ced::cpu;
    Point a = {0,0};// 0  
    Point b = {4,0};// 1 
    Point c = {4,4};// 2 
    Point d = {0,4};// 3 

    std::vector<Point> coords = {a, b, c, d};
    std::vector<unsigned int> triangles = {0,3,2,2,1,0};
    unsigned int amountOfTri = triangles.size()/3;
    unsigned int width = 5; 
    unsigned int height = 5;
    unsigned int amountOfPix = (width) *(height);
    ASSERT_EQ(amountOfPix, (unsigned int)25);
    std::multimap<unsigned int, unsigned int> pixelsWithTri;
    for(unsigned int t = 0; t < amountOfTri; ++t)
    {
        unsigned int idTri0 = triangles[(t*3+0)];
        unsigned int idTri1 = triangles[(t*3+1)];
        unsigned int idTri2 = triangles[(t*3+2)];
        Point v0 = coords[idTri0];
        Point v1 = coords[idTri1];
        Point v2 = coords[idTri2];
        for(unsigned int p = 0; p < amountOfPix; ++p)
        {
            unsigned int h = p % width;
            unsigned int w = p / width;
            Point pix = {w, h};   
            if(isPixInTri(v0, v1, v2, pix))
            {
                pixelsWithTri.insert({t, p});
            }
        }
    }
    
    // this is probably another test
    std::vector<unsigned int> ids;

    for(unsigned int t = 0 ; t < 2; ++t)
    {
        ids.clear();
        for(auto const& x : pixelsWithTri)
        {
            if(t == x.first)
            {
                ids.push_back(x.second);
            } 
        }
        // do something 
    }


   EXPECT_EQ(ids[0], 0);
   EXPECT_EQ(ids[1], 5);
   EXPECT_EQ(ids[2], 6);
   EXPECT_EQ(ids[3], 10);
   EXPECT_EQ(ids[4], 11);

   EXPECT_EQ(ids[5], 12);
   EXPECT_EQ(ids[6], 15);
   EXPECT_EQ(ids[7], 16);
   EXPECT_EQ(ids[8], 17);
   EXPECT_EQ(ids[9], 18);

   EXPECT_EQ(ids[10], 20);
   EXPECT_EQ(ids[11], 21);
   EXPECT_EQ(ids[12], 22);
   EXPECT_EQ(ids[13], 23);
   EXPECT_EQ(ids[14], 24);
}
