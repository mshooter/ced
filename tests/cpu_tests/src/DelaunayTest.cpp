#include "gtest/gtest.h"

#include "Point.hpp"
#include "Edge.hpp"
#include "Triangle.hpp"
#include "isLeft.hpp"

#include <iostream>

TEST(Delaunay, checkIfPointIsLeft)
{
    using namespace ced::cpu;
    Point* p1 = new Point(0,0);
    Point* p2 = new Point(1,1);
    Point* p3 = new Point(2,2);

    std::vector<Point*> pointList = {p1, p2, p3};
    std::vector<Triangle*> triangleList; 
    std::vector<Edge*> eHull;
    // init the hull 
    int lOrR = ((p3->x - p1->x)*(p2->y - p1->y)) - ((p2->x - p1->x) * (p3->y - p1->y));
    ASSERT_EQ(lOrR, 0);
    // lOrR if true 
    // p2 is on the left side p0p1   
    if(lOrR <= 0)
    {
        triangleList.push_back(new Triangle(p1,p2,p3));
        eHull.push_back(new Edge(p1, p2)); 
        eHull.push_back(new Edge(p2, p3)); 
        eHull.push_back(new Edge(p3, p1)); 
        
    }

    EXPECT_EQ((int)triangleList.size(), 1);
    EXPECT_EQ((int)eHull.size(), 3);

    EXPECT_EQ(triangleList[0]->getVertices()[0]->x, 0); 
    EXPECT_EQ(triangleList[0]->getVertices()[1]->x, 1); 
    EXPECT_EQ(triangleList[0]->getVertices()[2]->x, 2); 

    EXPECT_EQ(eHull[0]->startPoint->x, 0);
    EXPECT_EQ(eHull[1]->startPoint->x, 1);
    EXPECT_EQ(eHull[2]->startPoint->x, 2);
    
    EXPECT_EQ(eHull[0]->endPoint->x, 1);
    EXPECT_EQ(eHull[1]->endPoint->x, 2);
    EXPECT_EQ(eHull[2]->endPoint->x, 0);
}

TEST(Delaunay, checkIfPointIsRight)
{
    using namespace ced::cpu;
    Point* p1 = new Point(0,0);
    Point* p2 = new Point(1,1);
    Point* p3 = new Point(4,2);

    std::vector<Point*> pointList = {p1, p2, p3};
    std::vector<Triangle*> triangleList; 
    std::vector<Edge*> eHull;
    // init the hull 
    int lOrR = ((p3->x - p1->x)*(p2->y - p1->y)) - ((p2->x - p1->x) * (p3->y - p1->y));
    ASSERT_EQ(lOrR, 2);
    // lOrR if true 
    // p2 is on the left side p0p1   
    if(lOrR > 0)
    {
        triangleList.push_back(new Triangle(p2,p1,p3));
        eHull.push_back(new Edge(p2, p1)); 
        eHull.push_back(new Edge(p1, p3)); 
        eHull.push_back(new Edge(p3, p2)); 
        
    }

    EXPECT_EQ((int)triangleList.size(), 1);
    EXPECT_EQ((int)eHull.size(), 3);

    EXPECT_EQ(triangleList[0]->getVertices()[0]->x, 1); 
    EXPECT_EQ(triangleList[0]->getVertices()[1]->x, 0); 
    EXPECT_EQ(triangleList[0]->getVertices()[2]->x, 4); 

    EXPECT_EQ(eHull[0]->startPoint->x, 1);
    EXPECT_EQ(eHull[1]->startPoint->x, 0);
    EXPECT_EQ(eHull[2]->startPoint->x, 4);
    
    EXPECT_EQ(eHull[0]->endPoint->x, 0);
    EXPECT_EQ(eHull[1]->endPoint->x, 4);
    EXPECT_EQ(eHull[2]->endPoint->x, 1);
}

TEST(Delaunay, addPoint)
{
    using namespace ced::cpu;
    Point* p1 = new Point(0,0);
    Point* p2 = new Point(1,1);
    Point* p3 = new Point(4,0);
    Point* p4 = new Point(-1,5);

    std::vector<Point*> pointList = {p1, p2, p3, p4};
    std::vector<Triangle*> triangleList; 
    std::vector<Edge*> eHull;
    
    bool l = isLeft(p1,p2,p3);    
    ASSERT_EQ(l, false);
    triangleList.push_back(new Triangle(p2, p1, p3)); 
    eHull.push_back(new Edge(p2, p1));
    eHull.push_back(new Edge(p1, p3));
    eHull.push_back(new Edge(p3, p2));
    // go over every point in the list
    for(unsigned int i = 3; i < pointList.size(); ++i)
    {
        Point* newPoint = pointList[i];
        // go over every point in the edge
        for(unsigned int j=0; j < eHull.size(); ++j)
        {
            Point* s_edgePoint = eHull[j]->startPoint;
            Point* e_edgePoint = eHull[j]->endPoint;
            if(!isLeft(s_edgePoint, e_edgePoint, newPoint))
            {
                // should be two edges = correct
                // std::cout<< "start"<<s_edgePoint->x  << " " << s_edgePoint->y<< std::endl; 
                // std::cout<< "end"<<e_edgePoint->x  << " " << e_edgePoint->y<< std::endl; 
                // base edge 
                int baseIndex = j;
                // need to remove base edge soon but first create the two edges
                Edge* e_13 = new Edge(s_edgePoint, newPoint);
                Edge* e_32 = new Edge(newPoint, e_edgePoint);
                // check if the reverse is in the hull
            }
        } 
    }
}

