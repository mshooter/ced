#include "gtest/gtest.h"
#include <algorithm>
#include "Point.hpp"
#include "TriOrientation.hpp"
#include "ColourTriangle.hpp"

TEST(AssignColourToTrianglesTest, computeBB)
{
    // get min x, y 
    using namespace ced::cpu;
    Point p =  {0,0}; 
    Point p1 = {1,1}; 
    Point p2 = {2,0}; 
    std::pair<int, int> minmaxx = std::minmax({p.x, p1.x, p2.x});
    std::pair<int, int> minmaxy = std::minmax({p.y, p1.y, p2.y});
    EXPECT_EQ(minmaxx.first, 0); // minx
    EXPECT_EQ(minmaxx.second, 2); // maxx
    EXPECT_EQ(minmaxy.first, 0); // miny
    EXPECT_EQ(minmaxy.second, 1); //maxy
    // screenn bounds too 
    minmaxx.first = std::max(minmaxx.first, 0); 
    // screen
    minmaxx.second = std::min(minmaxx.second, 100); 
    minmaxy.first = std::max(minmaxy.first, 0); 
    // screen
    minmaxy.second = std::min(minmaxy.second, 100); 
    EXPECT_EQ(minmaxx.first, 0); // minx
    EXPECT_EQ(minmaxx.second, 2); // maxx
    EXPECT_EQ(minmaxy.first, 0); // miny
    EXPECT_EQ(minmaxy.second, 1); //maxy
    int minx, maxx, miny, maxy;
    findMinMax(p, p1, p2, 100, 100, minx, maxx, miny, maxy);
}

TEST(AssignColourToTrianglesTest, Rasterize)
{
    using namespace ced::cpu;
    Point p0 =  {0,0}; 
    Point p1 = {1,3}; 
    Point p2 = {4,0}; 
    int minx = 0;
    int miny = 0;
    int maxx = 4;
    int maxy = 3;
    Point p;
    int amountOfPix =0;
    for(p.y = miny; p.y <= maxy; ++p.y)
    {
        for(p.x = minx; p.x <= maxx; ++p.x)
        {
            // barcentric coordinates
            int w0 = isCCW<int>(p1, p2, p); 
            int w1 = isCCW<int>(p2, p0, p); 
            int w2 = isCCW<int>(p0, p1, p); 
            // if p is on or inside all edges render pixel
            if(w0 >= 0 && w1 >= 0 && w2 >= 0)
            {
                // do something with pixel
                ++amountOfPix;
            }
        }
    }
    EXPECT_EQ(amountOfPix, 11);
    std::vector<Point> tripts;
}
