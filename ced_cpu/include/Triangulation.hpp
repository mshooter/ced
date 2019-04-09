#ifndef TRIANGULATION_H_INCLUDED
#define TRIANGULATION_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        //  -----------------------------------------------------------------------
        /// @build : triangulate 
        /// @param[_in] : list of points
        /// @return[_out] : list of points ordered in [p1, p2, p3] of a triangle;
        //  -----------------------------------------------------------------------
        void triangulate(std::vector<Point> const& _points);
        Point calculateCentroidCenter(std::vector<Point> _pts);
        void createFirstTri(std::vector<Point> pts, unsigned int& i0, unsigned int& i1, unsigned int& i2, Point centroid); 
        unsigned int hash_key(Point p, Point cc, unsigned int hashSize);
        /// @build : link the halfedges
        void link(const unsigned int idx1, const unsigned int idx2, std::vector<uint>& halfedges); 
        /// @build : vertex indices & adjacent half-edge ids
        unsigned int add_triangle(uint i0, uint i1, uint i2, uint h_i0, uint h_i1, uint h_i2, std::vector<uint>& triangles, std::vector<uint>& halfedges); 
    }
}

#endif // TRIANGULATION_H_INCLUDED
