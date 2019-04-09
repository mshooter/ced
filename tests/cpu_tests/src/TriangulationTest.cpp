#include "gtest/gtest.h"

#include "Triangulation.hpp"
#include "Distance2P.hpp"
#include "CircumCircle.hpp"

//  -------------------------------------------------------------------------
TEST(Triangulation, calculateSeedCenter)
{
    ced::cpu::Point p1 = {0,0};
    ced::cpu::Point p2 = {1,1};
    ced::cpu::Point p3 = {2,0};
    
    std::vector<ced::cpu::Point> verts = {p1, p2, p3};
    
    using namespace ced::cpu;
    
    Point p = calculateCentroidCenter(verts); 
    EXPECT_EQ(p.x, 1);
    EXPECT_EQ(p.y, 0.5f);

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
            r = circumRadius<float>(verts[i0], verts[i1], point); 
            if(r < min_radius )
            {
                i2 = i;
                min_radius = r;
            }
        } 
        ++i;
    }
    EXPECT_EQ(i2, (int)2);
    EXPECT_FLOAT_EQ(r, 0.0f);
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
    bool ccw = isCCW<float>(verts[i0], verts[i1], verts[i2]);
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
TEST(Triangulation, findSeedNearCentroid)
{
    using namespace ced::cpu;
    Point p1 = {0,0};
    Point p2 = {1,1};
    Point p3 = {2,0};
    
    std::vector<Point> _points = {p1, p2, p3};
    Point sc = calculateCentroidCenter(_points);
    ASSERT_EQ(Point(1.0f, 0.5f), sc);
    float min_dist = std::numeric_limits<float>::max();
    unsigned int i0 = INVALID_IDX; 
    unsigned int i1 = INVALID_IDX; 
    unsigned int i2 = INVALID_IDX; 
    // seed point close to centroid
    unsigned int i = 0;
    for(auto pt : _points)
    {
        const float d = distance2P<float>(sc, pt); 
        if(d < min_dist)
        {
            i0 = i; 
            min_dist = d;
        }
        ++i;
    }
    EXPECT_EQ(i0, (unsigned int)1);
    Point pi0 = _points[i0];
    EXPECT_EQ(pi0, Point(1,1));

    min_dist = std::numeric_limits<float>::max();
    i = 0;
    // find point close to seed
    for(auto pt : _points)
    {
        if(i != i0)
        {
            const float d = distance2P<float>(pi0, pt);
            if(d < min_dist && d > 0.0f)
            {
                i1 = i;
                min_dist = d;
            }
        }
        ++i;
    } 
    EXPECT_EQ(i1, (unsigned int)0);
    Point pi1 = _points[i1];
    EXPECT_EQ(pi1, Point(0,0));

    // third point
    float min_radius = std::numeric_limits<float>::max();
    i = 0;
    // find point close to seed
    for(auto pt : _points)
    {
        if(i != i0 && i != i1)
        {
            const float r  = circumRadius<float>(pi0, pi1, pt);
            if(r < min_radius)
            {
                i2 = i;
                min_radius = r;
            }
        }
        ++i;
    } 
    EXPECT_EQ(i2, (unsigned int)2);
    Point pi2 = _points[i2];
    EXPECT_EQ(pi2, Point(2,0));
    
    createFirstTri(_points, i0, i1, i2, sc); 
    EXPECT_EQ(i0, (unsigned int)1);
    EXPECT_EQ(i1, (unsigned int)0);
    EXPECT_EQ(i2, (unsigned int)2);
}
//  -------------------------------------------------------------------------
TEST(Triangulation, swapPoint)
{
    using namespace ced::cpu;
    int i1 = 1; 
    int i2 = 2; 
    Point p1 = {0,4};
    Point p2 = {3,3};
    std::swap(i1, i2);
    std::swap(p1, p2);
    EXPECT_EQ(i1, 2);
    EXPECT_EQ(p1, Point(3,3));
}
//  -------------------------------------------------------------------------
TEST(Triangulation, fillVect)
{
    std::vector<int> ids;
    ids.resize(3);
    std::fill(ids.begin(), ids.end(), 10);
    EXPECT_EQ(ids[0], 10);
    EXPECT_EQ(ids[1], 10);
    EXPECT_EQ(ids[2], 10); 
}
//  -------------------------------------------------------------------------
TEST(Triangulation, hashKey)
{
    using namespace ced::cpu;
    Point p = {0,0};
    Point cc = {2,0};
    EXPECT_EQ(hash_key(p,cc,3), (uint)0);
    
}
//  -------------------------------------------------------------------------
TEST(Triangulation, link)
{
    using namespace ced::cpu;
    std::vector<uint> halfedges;
    halfedges.reserve(3);
    // if invalid indx
    link(0, INVALID_IDX, halfedges);
    link(1, INVALID_IDX, halfedges);
    link(2, INVALID_IDX, halfedges);
    EXPECT_EQ(halfedges[0], (unsigned int)INVALID_IDX);
    EXPECT_EQ(halfedges[1], (unsigned int)INVALID_IDX);
    EXPECT_EQ(halfedges[2], (unsigned int)INVALID_IDX);
    EXPECT_EQ(halfedges.size(), (unsigned int)3);
}
//  -------------------------------------------------------------------------
TEST(Triangulation, addTriangle)
{
    using namespace ced::cpu;
    std::vector<uint> triangles;
    std::vector<uint> halfedges;
    triangles.reserve(3);
    halfedges.reserve(3);
    EXPECT_EQ(add_triangle(0, 1, 2, INVALID_IDX, INVALID_IDX, INVALID_IDX, triangles, halfedges), (uint)0);
}
//  -------------------------------------------------------------------------
#include <iostream>
TEST(Triangulation, skipDuplicates)
{
    using namespace ced::cpu; 
    std::vector<int> vert = {1,2,3,4}; 
    Point pp;
    for(auto& x : vert)
    {
        uint k = &x - &vert[0]; 
        const uint idx = x; 
        const Point p = _points[idx]; 
        // skip near duplicates
        // if point is not centroid/seed and if the points are equal -> skip
        if(k > 0 && equalPts(p, pp)) continue;
    }
    
}
//  -------------------------------------------------------------------------
