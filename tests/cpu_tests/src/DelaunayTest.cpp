#include "gtest/gtest.h"

#include "Point.hpp"
#include "Edge.hpp"
#include "Triangle.hpp"
#include "Delaunay.hpp"

#include <iostream>
// ------------------------------------------------------------------------------------------------------------
// 1) new point is selected 
// 2) iterate over all the edges in the hull 
// 3) are they facing the right side? -> if true: select that edge as the base edge
// 4) construct a triangle 
// 5) two edges are created 
// I DID THE SEARCHING THE PAPER SHOWS ANOTHER EXAMPLE -> implement that performance is better
// 6) check if the reversed version is in the hull, if it is -> remove the reversed version from the hull 
// 7) if false -> the upper and lower edge are inserted in the hull 
// ------------------------------------------------------------------------------------------------------------

TEST(Delaunay, checkIfPointIsLeft)
{
    using namespace ced::cpu;
    Point p1 = Point(0,0);
    Point p2 = Point(1,1);
    Point p3 = Point(2,2);

    std::vector<Point> pointList = {p1, p2, p3};
    std::vector<Triangle> triangleList; 
    std::vector<Edge> eHull;
    // init the hull 
    int lOrR = ((p3.x - p1.x)*(p2.y - p1.y)) - ((p2.x - p1.x) * (p3.y - p1.y));
    ASSERT_EQ(lOrR, 0);
    // lOrR if true 
    // p2 is on the left side p0p1   
    if(lOrR <= 0)
    {
        triangleList.push_back(Triangle(p1,p2,p3));
        eHull.push_back(Edge(p1, p2)); 
        eHull.push_back(Edge(p2, p3)); 
        eHull.push_back(Edge(p3, p1)); 
        
    }

    EXPECT_EQ((int)triangleList.size(), 1);
    EXPECT_EQ((int)eHull.size(), 3);

    EXPECT_EQ(triangleList[0].getVertices()[0].x, 0); 
    EXPECT_EQ(triangleList[0].getVertices()[1].x, 1); 
    EXPECT_EQ(triangleList[0].getVertices()[2].x, 2); 

    EXPECT_EQ(eHull[0].startPoint.x, 0);
    EXPECT_EQ(eHull[1].startPoint.x, 1);
    EXPECT_EQ(eHull[2].startPoint.x, 2);
    
    EXPECT_EQ(eHull[0].endPoint.x, 1);
    EXPECT_EQ(eHull[1].endPoint.x, 2);
    EXPECT_EQ(eHull[2].endPoint.x, 0);
}
//-----------------------------------------------------------------------------------------------------------------
TEST(Delaunay, checkIfPointIsRight)
{
    using namespace ced::cpu;
    Point p1 = Point(0,0);
    Point p2 = Point(1,1);
    Point p3 = Point(4,2);

    std::vector<Point> pointList = {p1, p2, p3};
    std::vector<Triangle> triangleList; 
    std::vector<Edge> eHull;
    // init the hull 
    int lOrR = ((p3.x - p1.x)*(p2.y - p1.y)) - ((p2.x - p1.x) * (p3.y - p1.y));
    ASSERT_EQ(lOrR, 2);
    // lOrR if true 
    // p2 is on the left side p0p1   
    if(lOrR > 0)
    {
        triangleList.push_back(Triangle(p2,p1,p3));
        eHull.push_back(Edge(p2, p1)); 
        eHull.push_back(Edge(p1, p3)); 
        eHull.push_back(Edge(p3, p2)); 
        
    }

    EXPECT_EQ((int)triangleList.size(), 1);
    EXPECT_EQ((int)eHull.size(), 3);

    EXPECT_EQ(triangleList[0].getVertices()[0].x, 1); 
    EXPECT_EQ(triangleList[0].getVertices()[1].x, 0); 
    EXPECT_EQ(triangleList[0].getVertices()[2].x, 4); 

    EXPECT_EQ(eHull[0].startPoint.x, 1);
    EXPECT_EQ(eHull[1].startPoint.x, 0);
    EXPECT_EQ(eHull[2].startPoint.x, 4);
    
    EXPECT_EQ(eHull[0].endPoint.x, 0);
    EXPECT_EQ(eHull[1].endPoint.x, 4);
    EXPECT_EQ(eHull[2].endPoint.x, 1);
}
//-----------------------------------------------------------------------------------------------------------------
TEST(Delauany, removeItem)
{
    using namespace ced::cpu;
    Point p1(0,1);
    Point p2(1,2);
    Edge e(p1, p2);
    Point p3(3,1);
    Point p4(4,2);
    Edge e2(p3, p4);
    std::vector<Edge> hull = {e,e2};
    // erase element from the hull
    hull.erase(std::remove(hull.begin(), hull.end(), Edge(p1,p2)), hull.end());
//   for(auto edge : hull)
//   {
//       std::cout<<"START "<<edge.startPoint.x << " " << edge.startPoint.y << std::endl;
//       std::cout<<"END "<<edge.endPoint.x << " " << edge.endPoint.y << std::endl;
//   }
//   std::cout<<"SIZE "<<hull.size();
    int sizeOfHull = hull.size();
    EXPECT_EQ(sizeOfHull, 1);
    EXPECT_EQ(hull[0].startPoint.x, 3);
    EXPECT_EQ(hull[0].startPoint.y, 1);
    EXPECT_EQ(hull[0].endPoint.x, 4);
    EXPECT_EQ(hull[0].endPoint.y, 2);

}
//-----------------------------------------------------------------------------------------------------------------
TEST(Delaunay, swapItems)
{
    using namespace ced::cpu;
    Point p1(0,1);
    Point p2(2,4);
    Edge edge(p1, p2);
    std::swap(edge.startPoint, edge.endPoint);
    EXPECT_EQ(edge.startPoint.x, 2);
    EXPECT_EQ(edge.endPoint.x, 0);

}
//-----------------------------------------------------------------------------------------------------------------
TEST(Delaunay, insertElement)
{

    using namespace ced::cpu;
    Point p1(3,0);
    Point p2(1,2);

    Point p3(4,0);
    Point p4(6,2);
    Edge e(p3, p4);
    std::vector<Edge> ehull;
    //std::cout<<"BEFORE size of ehull"<<ehull.size()<<std::endl;
    ehull.insert(ehull.begin(), Edge(p1, p2));
    insertBeforeElement(e, ehull.begin(), ehull);
    
    ASSERT_EQ(ehull[0].startPoint.x, 4);
    ASSERT_EQ(ehull[1].startPoint.x, 3);
    //std::cout<<"AFTER size of ehull"<<ehull.size()<<std::endl;
    
    // creating a dummy hull with integers to see if i can insert elements
    std::vector<int> hull = {1, 2, 3};
    auto idx = hull.end() ;
//  if(idx != hull.end())
//  {
//      newIt = hull.insert(idx +1, 4);
//  }
//  else
//  {
//      newIt = hull.insert(idx, 4);
//  }
    auto newIt = insertAfterElement(4, idx, hull);
    EXPECT_EQ(hull[3], 4);
    // insert an element before the iterator
    insertBeforeElement(5, newIt, hull);
    EXPECT_EQ(hull[2], 5);

//   for(auto x : hull)
//   {
//       std::cout<<x<<std::endl;
//   }
    
}
//-----------------------------------------------------------------------------------------------------------------
TEST(Delaunay, triangulate)
{
    using namespace ced::cpu;
    Point p0(0,0);
    Point p1(1,1);
    Point p2(2,0);

    Point p3(3,1);
    Point p4(4,-1);
    Point p5(6,0);
    std::vector<Point> pts = {p0, p1, p2, p3, p4, p5};
    std::vector<Triangle> tri;
    triangulate(pts, tri);
    //std::cout<<tri.size()<<" SIZE";
    EXPECT_EQ(tri[0], Triangle(p1, p0, p2)); 
    EXPECT_EQ(tri[1], Triangle(p1, p2, p3)); 
    EXPECT_EQ(tri[2], Triangle(p2, p0, p4)); 
    EXPECT_EQ(tri[3], Triangle(p2, p4, p5)); 
//  for(auto x : tri)
//  {
//      for(auto y : x.getVertices())
//      {
//          std::cout<<y.x<< " "<<y.y << std::endl;
//      }
//  }
}

