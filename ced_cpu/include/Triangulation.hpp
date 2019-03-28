#ifndef TRIANGULATION_H_INCLUDED
#define TRIANGULATION_H_INCLUDED

#include <vector>
#include <Point.hpp>

namespace ced
{
    namespace cpu
    {
        //  -----------------------------------------------------------------------
        /// @build : add triangle to the triangle list
        /// @param[_in] _id0 : index v0
        /// @param[_in] _id1 : index v1
        /// @param[_in] _id2 : index v2
        /// @param[_in] _ahe0 : index of adjacent half-edge ids
        /// @param[_in] _ahe1 : index of adjacent half-edge ids
        /// @param[_in] _ahe2 : index of adjacent half-edge ids
        //  -----------------------------------------------------------------------
        template <typename T>
        T addTriangle(T _id0, T _id1, T _id2, T _ahe0, T _ahe1, T _ahe2, std::vector<T>& _triangles, std::vector<T>& halfedges);
        //  -----------------------------------------------------------------------
        /// @build : triangulate 
        /// @param[_in] : list of points
        /// @return[_out] : list of points ordered in [p1, p2, p3] of a triangle;
        //  -----------------------------------------------------------------------
        void triangulate(std::vector<Point>& _points);
        //  -----------------------------------------------------------------------
        /// @build : calculate the hash key 
        //  -----------------------------------------------------------------------
        template <typename T, typename U>
        T hash_key(const U _point, const U _cc, T _hashSize);
        //  -----------------------------------------------------------------------
        /// @build : calculate pseudo angle
        //  -----------------------------------------------------------------------
        template <typename T, typename U>
        T pseudo_angle(const U _point);
        //  -----------------------------------------------------------------------
        /// @build : link the adjacent halfedges
        //  -----------------------------------------------------------------------
        void link(int _triangleIndex, int _idHalfEdge, std::vector<int>& halfedges);
        //  -----------------------------------------------------------------------
        /// @build : template implementations
        //  -----------------------------------------------------------------------
        #include "Triangulation.inl"
    }
}

#endif // TRIANGULATION_H_INCLUDED
