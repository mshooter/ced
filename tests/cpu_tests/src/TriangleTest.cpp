#include "gtest/gtest.h"
#include "Triangle.hpp"
#include <iostream>

TEST(Triangle, addVertex)
{
    ced::cpu::Triangle tri; 
    ced::cpu::Point* p = new ced::cpu::Point(2,3);
    tri.addVertex(p);

    EXPECT_EQ(tri.getVertices()[0]->x, 2);
    EXPECT_EQ(tri.getVertices()[0]->y, 3);
}


TEST(Triangle, addTriangle)
{
    ced::cpu::Point* p = new ced::cpu::Point(2,3);
    ced::cpu::Point* p1 = new ced::cpu::Point(21,3);
    ced::cpu::Point* p2 = new ced::cpu::Point(3,3);
    std::vector<ced::cpu::Point*> pts;

    pts.push_back(p);
    pts.push_back(p1);
    pts.push_back(p2);

    ced::cpu::Triangle* tri = new ced::cpu::Triangle(pts);  
    ced::cpu::Triangle triF;  
    triF.addTriangle(tri);

    auto triangleVerts = triF.getNeighbourTriangles()[0]->getVertices();
    EXPECT_EQ(triangleVerts[0]->x, 2);
    EXPECT_EQ(triangleVerts[0]->y, 3);
    EXPECT_EQ(triangleVerts[1]->x, 21);
    EXPECT_EQ(triangleVerts[2]->x, 3);
}

TEST(Triangle, addPoints)
{
    ced::cpu::Point* p = new ced::cpu::Point(2,3);
    ced::cpu::Point* p1 = new ced::cpu::Point(21,4);
    ced::cpu::Point* p2 = new ced::cpu::Point(5,1);
    
    ced::cpu::Triangle tri(p, p1, p2);

    EXPECT_EQ(tri.getVertices()[0]->x, 2);
    EXPECT_EQ(tri.getVertices()[0]->y, 3);

    EXPECT_EQ(tri.getVertices()[1]->x, 21);
    EXPECT_EQ(tri.getVertices()[1]->y, 4);

    EXPECT_EQ(tri.getVertices()[2]->x, 5);
    EXPECT_EQ(tri.getVertices()[2]->y, 1);
}
