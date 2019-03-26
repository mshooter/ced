#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>

#include "SortPoints.hpp"

struct vert
{
    int x; 
    int y;
};
//--------------------------------------------------------------------------------
// frontier = double connected edge list 
// linked list of triangles with all the necessary local information
    // needs three vertices
    // needs three edges
    // three pointers to its neighbouring triangles
// hash table on a circular duble linked list is used 
#include <vector>
struct triangle
{

    std::vector<int> v_indices;
};
//--------------------------------------------------------------------------------
bool compareX(const vert& lhs, const vert& rhs)
{
    return lhs.x < rhs.x;
}
//--------------------------------------------------------------------------------
bool compareY(const vert& lhs, const vert& rhs)
{
    return lhs.y < rhs.y;
}
//--------------------------------------------------------------------------------
TEST(dd, vert)
{
    // advantage
    vert v{1,0};
    EXPECT_EQ(1,v.x);     
    EXPECT_EQ(0,v.y);     
}
//--------------------------------------------------------------------------------
TEST(dd, vectorOfverts)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,1}, vert{3,0}};
    EXPECT_EQ(verts[0].x, 0);
    EXPECT_EQ(verts[1].x, 2);
    EXPECT_EQ(verts[2].x, 3);
}
//--------------------------------------------------------------------------------
TEST(dd, triangle)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,0}, vert{3,1}, vert{4,0}};
    triangle t; 
    t.v_indices.insert(t.v_indices.end(),{0, 1, 2});
    int idx0 = t.v_indices[0];
    int idx1 = t.v_indices[1];
    int idx2 = t.v_indices[2];
    EXPECT_EQ(verts[idx0].x, 0);
    EXPECT_EQ(verts[idx1].x, 2);
    EXPECT_EQ(verts[idx2].x, 3);
}
//--------------------------------------------------------------------------------
TEST(dd, minMaxX)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,0}, vert{3,1}, vert{4,0}};
    const auto [min, max] = std::minmax_element(verts.begin(), verts.end(), compareX);
    EXPECT_EQ(min->x, 0);
    EXPECT_EQ(min->y, 1);
    EXPECT_EQ(max->x, 4);
    EXPECT_EQ(max->y, 0);
}
//--------------------------------------------------------------------------------
TEST(dd, originPolarCoordPx)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,0}, vert{3,1}, vert{4,0}};
    const auto [min, max] = std::minmax_element(verts.begin(), verts.end(), compareX);
    ASSERT_EQ(min->x, 0);
    ASSERT_EQ(min->y, 1);
    ASSERT_EQ(max->x, 4);
    ASSERT_EQ(max->y, 0);
    int amountOfPoints = verts.size();
    int avg = (min->x + max->x) / amountOfPoints;  
    EXPECT_EQ(avg, 1);
}
//--------------------------------------------------------------------------------
TEST(dd, minMaxY)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,0}, vert{3,1}, vert{4,0}};
    const auto [min, max] = std::minmax_element(verts.begin(), verts.end(), compareY);
    EXPECT_EQ(min->x, 2);
    EXPECT_EQ(min->y, 0);
    EXPECT_EQ(max->x, 3);
    EXPECT_EQ(max->y, 1);
}
//--------------------------------------------------------------------------------
TEST(dd, originPolarCoordPy)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,0}, vert{3,1}, vert{4,0}};
    const auto [min, max] = std::minmax_element(verts.begin(), verts.end(), compareY);
    ASSERT_EQ(min->x, 2);
    ASSERT_EQ(min->y, 0);
    ASSERT_EQ(max->x, 3);
    ASSERT_EQ(max->y, 1);
    int amountOfPoints = verts.size();
    int avg = (min->y + max->y) / amountOfPoints;  
    EXPECT_EQ(avg, 0);
}
//--------------------------------------------------------------------------------
TEST(dd, polarCoordinates)
{
    // (x_i - origin) * 2
    vert o = {1,0};
    vert v_i = {0,1};
    int r = std::sqrt((v_i.x - o.x) * (v_i.x - o.x) + (v_i.y - o.y) * (v_i.y - o.y));
    int theta = 0;
    if((v_i.y - o.y) > 0)
    {
        theta = std::acos((v_i.x - o.x) / r);
    }
    else
    {
        theta = std::acos((v_i.x - o.x) / r) + 3.14f;
    }
    ASSERT_EQ(r, 1);
    ASSERT_EQ(theta, 3);
    v_i.x = r;
    v_i.y = theta;

    EXPECT_EQ(v_i.x, 1);
    EXPECT_EQ(v_i.y, 3);
}
//--------------------------------------------------------------------------------
//#include <iostream>
TEST(dd, sortPoints)
{
    std::vector<vert> verts = {vert{0,1}, vert{2,0}, vert{3,1}, vert{4,0}};
    vert o = {1,0};
    for(auto& v_i : verts)
    {
        int r = std::sqrt((v_i.x - o.x) * (v_i.x - o.x) + (v_i.y - o.y) * (v_i.y - o.y));

        if((v_i.y - o.y)> 0)
        {
            // theta
            v_i.y = std::acos((v_i.x - o.x) / r);
        }
        else
        {
            // theta
            v_i.y = std::acos((v_i.x - o.x) / r) + 3.14f;
        }
        v_i.x = r;
    }  

    ASSERT_EQ(verts[0].x, 1);
    ASSERT_EQ(verts[0].y, 3);

    ASSERT_EQ(verts[1].x, 1);
    ASSERT_EQ(verts[1].y, 3);  

    ASSERT_EQ(verts[2].x, 2);  
    ASSERT_EQ(verts[2].y, 0); 

    ASSERT_EQ(verts[3].x, 3);  
    ASSERT_EQ(verts[3].y, 3); 
    
    ced::cpu::quickSort(verts, 0, 3);

    EXPECT_EQ(verts[0].x, 1);
    EXPECT_EQ(verts[0].y, 3);

    EXPECT_EQ(verts[1].x, 1);
    EXPECT_EQ(verts[1].y, 3);

    EXPECT_EQ(verts[2].x, 2);
    EXPECT_EQ(verts[2].y, 0);

    EXPECT_EQ(verts[3].x, 3);
    EXPECT_EQ(verts[3].y, 3);
}
//--------------------------------------------------------------------------------
