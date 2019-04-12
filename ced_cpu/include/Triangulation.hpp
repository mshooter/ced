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
        void triangulate(std::vector<Point> const& _points, std::vector<unsigned int>& triangles);
        //  -----------------------------------------------------------------------
        /// @build : calculate the centroid center
        /// @param [_in] : list of points
        //  -----------------------------------------------------------------------
        Point calculateCentroidCenter(std::vector<Point> _pts);
        //  -----------------------------------------------------------------------
        /// @build : create the first triangle
        /// @param[_in] : list of points
        /// @param[_in] : index of first point
        /// @param[_in] : index of second point
        /// @param[_in] : index of third point
        /// @param[_in] : centroid point
        //  -----------------------------------------------------------------------
        void createFirstTri(    std::vector<Point> pts, 
                                unsigned int& i0, 
                                unsigned int& i1, 
                                unsigned int& i2, 
                                Point centroid); 
        //  -----------------------------------------------------------------------
        /// @build : calculate the hash key for the hash table 
        /// @param[_in] : current point
        /// @param[_in] : circum center point
        /// @param[_in] : the size of the hash table
        //  -----------------------------------------------------------------------
        unsigned int hash_key(Point p, Point cc, unsigned int hashSize);
        //  -----------------------------------------------------------------------
        /// @build : link the halfedges
        /// @param[_in] : index of start point halfedge
        /// @param[_in] : index of end point halfedge
        /// @param[_in] : list of halfedge indexes
        //  -----------------------------------------------------------------------
        void link(const unsigned int idx1, const unsigned int idx2, std::vector<unsigned int>& halfedges); 
        //  -----------------------------------------------------------------------
        /// @build : vertex indices & adjacent half-edge ids
        /// @param[_in] : triangle index points
        /// @param[_in] : triangle index points
        /// @param[_in] : triangle index points
        //
        /// @param[_in] : halfedge index points
        /// @param[_in] : halfedge index points
        /// @param[_in] : halfedge index points
        //
        /// @param[_in] : triangle index list
        /// @param[_in] : halfedge index list
        //  -----------------------------------------------------------------------
        unsigned int add_triangle(  unsigned int i0, 
                                    unsigned int i1, 
                                    unsigned int i2, 
                                    unsigned int h_i0, 
                                    unsigned int h_i1, 
                                    unsigned int h_i2, 
                                    std::vector<unsigned int>& triangles, 
                                    std::vector<unsigned int>& halfedges); 
        //  -----------------------------------------------------------------------
        /// @build : check if it is delaunay
        /// @param[_in] : edge stack index list
        /// @param[_in] : triangle index list
        /// @param[_in] : halfedge index list
        /// @param[_in] : point list
        /// @param[_in] : index where the hull start
        /// @param[_in] : index list of hull (next points)
        /// @param[_in] : index list of hull (prev points)
        //  -----------------------------------------------------------------------
        unsigned int legalise(  unsigned int a,
                                std::vector<unsigned int>& edge_stack,
                                std::vector<unsigned int> triangles,
                                std::vector<unsigned int> halfedges,
                                std::vector<unsigned int>& hull_next, 
                                std::vector<unsigned int>& hull_tri,
                                std::vector<Point> pts,
                                unsigned int& hull_start);
        //  -----------------------------------------------------------------------
    }
}

#endif // TRIANGULATION_H_INCLUDED
