#include "Delaunay.hpp"
#include "Triangle.hpp"
#include "Edge.hpp"
namespace ced
{
    namespace cpu
    {
        //  --------------------------------------------------------------------------
        bool isLeft(Point p1, Point p2, Point p3)
        {
            int v = ((p3.x - p1.x) * (p2.y - p1.y)) - ((p2.x - p1.x) * (p3.y - p1.y));
            // is left
            if( v <= 0 )    { return true;  }
            // is right
            else            { return false; }
        }
        //  --------------------------------------------------------------------------
        void triangulate(std::vector<Point> _points)
        {
            // triangeList should be in the function referenced
            std::vector<Triangle> triangleList;
            // hull should be in the function referenced
            std::vector<Edge> hull;
            // create the first triangle
            // you need to check if the third point is on the right
            Point p0 = _points[0];
            Point p1 = _points[1];
            Point p2 = _points[2];
            if(!isLeft(p0, p1, p2))
            {
                // create a triangle 
                // insert that triangle in the list
                triangleList.push_back(Triangle(p1, p0, p2));
                hull.push_back(Edge(p1, p0));
                hull.push_back(Edge(p0, p2));
                hull.push_back(Edge(p2, p1));
                
            }
            else
            {
                triangleList.push_back(Triangle(p0, p1, p2));
                hull.push_back(Edge(p0, p1));
                hull.push_back(Edge(p1, p2));
                hull.push_back(Edge(p2, p0));
            }
            // iterate over the rest of the points
            uint sizeOfPoints = _points.size();
            for(uint i=3; i < sizeOfPoints; ++i)
            {
                // new point to add
                Point p = _points[i];
                for(auto edge : hull)
                {
                    // the start and end point of my edge
                    Point s = edge.startPoint;
                    Point e = edge.endPoint;
                    // check if the new point is on the right of the edge
                    if(!isLeft(s, e, p))
                    {
                        Edge baseEdge = edge;
                        // get iterator 
                        Edge lowSp = Edge(s, p); 
                        Edge upperPe = Edge(p, e); 
                        triangleList.push_back(Triangle(p1, p0, p2));
                        // hull check if the reverse is in there 
                        // need to create a test for that
                        if(isReverseEdgeInHull(lowSp, hull))
                        {
                            // remove the edge
                        }
                        else
                        {
                            // insert before
                        }

                        if(isReverseEdgeInHull(upperPe, hull))
                        {
                            // remove the edge
                        }
                        else
                        {
                            // insert after
                        }
                    }
                }
            }

        }
    }
}
