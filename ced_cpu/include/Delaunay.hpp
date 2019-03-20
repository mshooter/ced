#ifndef DELAUNAY_H_INCLUDED
#define DELAUNAY_H_INCLUDED

#include <vector>
#include <algorithm>
#include "Point.hpp"
#include "Triangle.hpp"
#include "Edge.hpp"

// TODO: create class DelaunayTri because it is a big class and you do not want all the functions public
namespace ced
{
    namespace cpu
    {

        //  --------------------------------------------------------------------------
        /// @build : checks if the third point is left from the first and second point
        /// the first point is mostly the startpoint of the edge and idem for last point ( end point)
        /// @param[_in] _p1 : first point (start point of edge)
        /// @param[_in] _p2 : second point (end point of edge)
        /// @param[_in] _p3 : third point (point to be added to form a triangle)
        //  --------------------------------------------------------------------------
        bool isLeft(Point _p1, Point _p2, Point _p3);
        //  --------------------------------------------------------------------------
        /// @build : checks if the third point is left from the first and second point
        /// the first point is mostly the startpoint of the edge and idem for last point ( end point)
        /// @param[_in] _x0 : x point of start point
        /// @param[_in] _x1 : x point of end point
        /// @param[_in] _x2 : x point of new point
        //
        /// @param[_in] _y0 : y point of start point
        /// @param[_in] _y1 : y point of end point
        /// @param[_in] _y2 : y point of new point
        //  --------------------------------------------------------------------------
        template <typename T>
        bool pointIsLeft(T x_0, T x_1, T x_2, T y_0, T y_1, T y_2);
        //  --------------------------------------------------------------------------
        /// @build : checks if the reverse of an edge is in the hull
        /// @param[_in] _edge : the edge to check if its reverse is in the hull
        /// @param[_in] _hull : the list of hull
        //  --------------------------------------------------------------------------
        template <typename T>
        bool isReverseEdgeInHull(T _edge, std::vector<T> _hull);
        //  --------------------------------------------------------------------------
        /// @build : inserting edge infront 
        /// @param[_in] _element : element to insert in the hull
        /// @param[_in] _bei : base edge index 
        //  --------------------------------------------------------------------------
        template <typename T>
        void insertBeforeElement(T _element, typename std::vector<T>::iterator _bei, std::vector<T>& _hull);
        //  --------------------------------------------------------------------------
        /// @build : inserting edge  after element
        /// @param[_in] _element : element to insert in the hull
        /// @param[_in] _bei : base edge index 
        //  --------------------------------------------------------------------------
        template <typename T>
        typename std::vector<T>::iterator insertAfterElement(T _element, typename std::vector<T>::iterator _bei, std::vector<T>& _hull);
        //  --------------------------------------------------------------------------
        /// @build : need a delaunay function that iterates over the rest of the points
        /// @param[_in] _points : the points are already sorted
        //  --------------------------------------------------------------------------
        void triangulate(std::vector<Point> _points, std::vector<Triangle>& _triangles);
        //  --------------------------------------------------------------------------
        /// @build : inserting the delaunay template functions
        //  --------------------------------------------------------------------------
        #include "Delaunay.inl"
    }
}

#endif //DELAUNAY_H_INCLUDED
