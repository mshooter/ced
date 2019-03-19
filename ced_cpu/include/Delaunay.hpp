#ifndef DELAUNAY_H_INCLUDED
#define DELAUNAY_H_INCLUDED

#include <vector>
#include <algorithm>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        //--------------------------------------------------------------------------
        /// @build : checks if the reverse of an edge is in the hull
        /// @param[_in] _edge : the edge to check if its reverse is in the hull
        /// @param[_in] _hull : the list of hull
        ///--------------------------------------------------------------------------
        template <typename T>
        bool isReverseEdgeInHull(T _edge, std::vector<T> _hull);
        //--------------------------------------------------------------------------
        /// @build : checks if the third point is left from the first and second point
        /// the first point is mostly the startpoint of the edge and idem for last point ( end point)
        /// @param[_in] _p1 : first point (start point of edge)
        /// @param[_in] _p2 : second point (end point of edge)
        /// @param[_in] _p3 : third point (point to be added to form a triangle)
        //--------------------------------------------------------------------------
        bool isLeft(Point _p1, Point _p2, Point _p3);

        #include "Delaunay.inl"
    }
}

#endif //DELAUNAY_H_INCLUDED
