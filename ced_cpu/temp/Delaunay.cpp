#include "Delaunay.hpp"
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
        void triangulate(std::vector<Point> _points, std::vector<Triangle>& _triangles)
        {
            // create a hull
            std::vector<Edge> hull; 
            std::vector<Edge> connectedEdge;
            // create the first triangle
            // you need to check if the third point is on the right
            Point p0 = std::move(_points[0]);
            Point p1 = std::move(_points[1]);
            Point p2 = std::move(_points[2]);
            if(!isLeft(p0, p1, p2))
            {
                // create a triangle 
                // insert that triangle in the list
                _triangles.push_back(Triangle(p1, p0, p2));
                hull.push_back(Edge(p1, p0));
                hull.push_back(Edge(p0, p2));
                hull.push_back(Edge(p2, p1));
                
            }
            else
            {
                _triangles.push_back(Triangle(p0, p1, p2));
                hull.push_back(Edge(p0, p1));
                hull.push_back(Edge(p1, p2));
                hull.push_back(Edge(p2, p0));
            }
            // iterate over the rest of the points
            uint sizeOfPoints = _points.size();
            uint sizeOfHull = hull.size();
            for(uint i=3; i < sizeOfPoints; ++i)
            {
                Point p = _points[i];
                for(uint eg=0; eg < sizeOfHull; ++eg)
                {
                    Point s = hull[eg].startPoint;
                    Point e = hull[eg].endPoint;
                    if(!isLeft(s, e, p))
                    {
                        // base edge element
                        Edge baseEdge = hull[eg];
                        // iterator index base edge 
                        std::vector<Edge>::iterator it = hull.begin() + eg;
                        std::vector<Edge>::iterator newIt = hull.begin() + eg;
                        // construct triangle
                        _triangles.push_back(Triangle(e, s, p));
                        // lower edge
                        Edge lowEdge(s, p);
                        // upper edge
                        Edge upperEdge(p, e);
                        // check if reverse is in Hull
                        if(isReverseEdgeInHull(upperEdge, hull))
                        {
                            // remove the reversed version of the hull
                            hull.erase(std::remove(hull.begin(), hull.end(), Edge(e,p)), hull.end());
                        }
                        else
                        {
                            // insert
                            newIt = insertAfterElement(upperEdge, it, hull); 
                        } 
                        if(isReverseEdgeInHull(lowEdge, hull))
                        {
                            hull.erase(std::remove(hull.begin(), hull.end(), Edge(p,s)), hull.end());
                        }
                        else
                        {
                            insertBeforeElement(lowEdge, newIt, hull);
                        }
                        connectedEdge.push_back(baseEdge); 
                        hull.erase(std::remove(hull.begin(), hull.end(), baseEdge), hull.end());
                    }
                } 
            }
            //return connectedEdge;
        }
        //  --------------------------------------------------------------------------
    }
}
