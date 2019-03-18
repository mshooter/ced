#include "gtest/gtest.h"

#include "Point.hpp"
#include "Edge.hpp"
#include "Triangle.hpp"
#include "Delaunay.hpp"

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
    eHull.push_back(new Edge(p2, p1)); // index 0 edge
    eHull.push_back(new Edge(p1, p3)); // index 1 edge
    eHull.push_back(new Edge(p3, p2)); // index 2 edge
    // go over every point in the list
    Point* newPoint = pointList[3];
    ASSERT_EQ(newPoint->x, -1);
    ASSERT_EQ(newPoint->y, 5);
    // go over every edge in the hull
    // the ones that are on the right 
    Point* s_edge = p2;
    Point* e_edge = p1;
    ASSERT_EQ(isLeft(s_edge, e_edge, newPoint), false); 
    // if it is on the right create a triangle 
    // record also the base edge 
    int baseIndex = 0;
    ASSERT_EQ(eHull[baseIndex]->startPoint->x, 1);
    ASSERT_EQ(eHull[baseIndex]->startPoint->y, 1);
    ASSERT_EQ(eHull[baseIndex]->endPoint->x, 0);
    ASSERT_EQ(eHull[baseIndex]->endPoint->y, 0);
    triangleList.push_back(new Triangle(e_edge, s_edge, newPoint));
    // check if the reverse is in the hull for upper and lower
    // new edges 
    Edge* sp = new Edge(s_edge, newPoint);
    Edge* ep = new Edge(newPoint, e_edge);
    Edge* r_sp = new Edge(newPoint, s_edge);
    Edge* r_ep = new Edge(e_edge, newPoint);
    //eHull.push_back(r_sp);
    ASSERT_EQ(isReverseInHull(r_sp, eHull), false);  
   
//   bool upper = false; 
//   bool lower = false;
//   for(auto edge : eHull)
//   {
//       if(r_sp == edge)
//       {
//           upper = true;
//       }
//   }
//    ASSERT_EQ(upper, false);
//  eHull.insert(eHull.begin() + (baseIndex+1), sp);
//  ASSERT_EQ(eHull[1]->startPoint->x, 1);
//  ASSERT_EQ(eHull[1]->endPoint->x, -1);
//  for(auto edge : eHull)
//  {
//      if(r_ep == edge)
//      {
//          lower = true;
//      }
//  }
//  ASSERT_EQ(lower, false);
//  // 
//  eHull.insert(eHull.begin() + baseIndex, ep);
//  baseIndex++;
//  ASSERT_EQ(eHull[0]->startPoint->x, -1);
//  ASSERT_EQ(eHull[0]->endPoint->x, 0);
//  eHull.erase(eHull.begin() + baseIndex);
    //for(auto edge : eHull)
    //{
    //    std::cout<<edge->startPoint->x<< " " << edge->startPoint->y << " "<<  edge->endPoint->x<< " " << edge->endPoint->y << std::endl;
    //}
}

