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
}
