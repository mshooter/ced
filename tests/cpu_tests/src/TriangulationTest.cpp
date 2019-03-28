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

   int i0;
   float minDistance = std::numeric_limits<float>::max();
   int i = 0;
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
   EXPECT_EQ(i0, (int)1);
 
}
//  -------------------------------------------------------------------------
TEST(Triangulation, findPointCloseToSeed)
{
    ced::cpu::Point p1 = {0,0};
    ced::cpu::Point p2 = {1,1};
    ced::cpu::Point p3 = {2,0};
    
    std::vector<ced::cpu::Point> verts = {p1, p2, p3};
    
    int i0 = 1;
    int i1;
    ASSERT_FLOAT_EQ(verts[i0].x, 1.0f); 
    ASSERT_FLOAT_EQ(verts[i0].y, 1.0f); 

    float minDistance = std::numeric_limits<float>::max();
    int i=0;
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

    EXPECT_EQ(i1, (int)0);
    
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
    int i0 = 1; 
    int i1 = 0; 
    int i2;
    
    float min_radius = std::numeric_limits<double>::max();
    float r;

    int i = 0;
    for(auto point : verts)
    {       
        // if i is not equal to 0 or 1 
        if(i != i0 && i != i1)
        { 
            //std::cout<<"POINT"<<point.x<<point.y<<std::endl;
            r = circumRadius<Point, float>(verts[i0], verts[i1], point); 
            if(r < min_radius )
            {
                i2 = i;
                min_radius = r;
            }
        } 
        ++i;
    }
    EXPECT_EQ(i2, (int)2);
    EXPECT_FLOAT_EQ(r, 1.0f);
}
//  -------------------------------------------------------------------------
#include "TriOrientation.hpp"
TEST(Triangulation, isCWW)
{
    using namespace ced::cpu;
    Point p1 = {0,0};
    Point p2 = {1,1};
    Point p3 = {2,0};
    
    std::vector<Point> verts = {p1, p2, p3};
    int i0 = 1;
    int i1 = 0;
    int i2 = 2;
    bool ccw = isCCW<Point, float>(verts[i0], verts[i1], verts[i2]);
    ASSERT_TRUE(ccw);
    
    if(ccw)
    {
        std::swap(i1, i2);
        std::swap(verts[i1], verts[i2]);
    }
    
    EXPECT_EQ(i0, (int)1);
    EXPECT_EQ(i1, (int)2);
    EXPECT_EQ(i2, (int)0);

    EXPECT_EQ(verts[0], p3);
    EXPECT_EQ(verts[1], p2);
    EXPECT_EQ(verts[2], p1);

}
//  -------------------------------------------------------------------------
#include <iostream>
TEST(Triangulation, ccCenter)
{
    using namespace ced::cpu;
    Point p1 = {0,0};
    Point p2 = {1,1};
    Point p3 = {2,0};
    Point cc = circumCenter<Point>(p3, p2, p1); 
    EXPECT_EQ(cc.x, 1);
    EXPECT_EQ(cc.y, 0);
}
//  -------------------------------------------------------------------------
TEST(Triangulation, calculateHashSizeCompare)
{  
    int hashSize = static_cast<int>(std::sqrt(2)); 
    if(hashSize < std::sqrt(2)) ++hashSize;
    
    EXPECT_EQ(hashSize, (int)2);
  
}
//  -------------------------------------------------------------------------
TEST(Triangulation, pseudoAngle)
{
    using namespace ced::cpu;
    Point p = {0,0};
    Point cc = {1,0};
    Point deltap = (p-cc);
    ASSERT_EQ(deltap, Point(-1,0));
    float tetha = pseudo_angle<float, Point>(deltap);
    ASSERT_FLOAT_EQ(tetha, 0);
}
///  -------------------------------------------------------------------------
TEST(Triangulation, hashkey)
{
    using namespace ced::cpu;
    Point pi0 = {1,1};
    Point pi1 = {2,0};
    Point pi2 = {0,0};
    Point cc  = {1,0};
    int key_1 = hash_key<int, Point>(pi0, cc, 2);  
    EXPECT_EQ(key_1, (int)1);
 
    int key_2 = hash_key<int, Point>(pi1, cc, 2);  
    EXPECT_EQ(key_2, (int)1);
    
    int key_3 = hash_key<int, Point>(pi2, cc, 2);  
    EXPECT_EQ(key_3, (int)0);
}
///  -------------------------------------------------------------------------
TEST(Triangulation, addTriangle)
{
    using namespace ced::cpu;
    std::vector<int> triangles;
    std::vector<int> halfedges;
    unsigned int max_tri = (2 * 3 - 5);
    triangles.reserve(max_tri * 3);
    halfedges.reserve(max_tri * 3);
    int i0 = 1;  
    int i1 = 2;  
    int i2 = 0;  

    int a = -1;
    int b = -1;
    int c = -1;

    EXPECT_EQ(triangles.size(), (unsigned int)0); 
    int t = addTriangle(i0, i1, i2, a, b, c, triangles, halfedges);
    
    EXPECT_EQ(triangles[0], 1); 
    EXPECT_EQ(triangles[1], 2); 
    EXPECT_EQ(triangles[2], 0); 
    
    EXPECT_EQ(halfedges[0], -1); 
    EXPECT_EQ(halfedges[1], -1); 
    EXPECT_EQ(halfedges[2], -1); 
    
    EXPECT_EQ(t, 3); 
}
///  -------------------------------------------------------------------------
TEST(Triangulation, duplicates)
{
    float x = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(x == 3);
}
///  -------------------------------------------------------------------------

