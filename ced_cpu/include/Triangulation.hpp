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
        void triangulate(std::vector<Point> const& _points, std::vector<int>& triangles);
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
                                int& i0, 
                                int& i1, 
                                int& i2, 
                                Point centroid); 
        //  -----------------------------------------------------------------------
        /// @build : calculate the hash key for the hash table 
        /// @param[_in] : current point
        /// @param[_in] : circum center point
        /// @param[_in] : the size of the hash table
        //  -----------------------------------------------------------------------
        int hash_key(Point p, Point cc, int hashSize);
        //  -----------------------------------------------------------------------
        /// @build : link the halfedges
        /// @param[_in] : index of start point halfedge
        /// @param[_in] : index of end point halfedge
        /// @param[_in] : list of halfedge indexes
        //  -----------------------------------------------------------------------
        void link(const int idx1, const int idx2, std::vector<int>& halfedges); 
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
        int add_triangle(  int i0, 
                                    int i1, 
                                    int i2, 
                                    int h_i0, 
                                    int h_i1, 
                                    int h_i2, 
                                    std::vector<int>& triangles, 
                                    std::vector<int>& halfedges); 
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
        int legalise(  int a,
                                std::vector<int>& edge_stack,
                                std::vector<int> triangles,
                                std::vector<int> halfedges,
                                std::vector<int>& hull_next, 
                                std::vector<int>& hull_tri,
                                std::vector<Point> pts,
                                int& hull_start);
        //  -----------------------------------------------------------------------
    }
}

#endif // TRIANGULATION_H_INCLUDED
