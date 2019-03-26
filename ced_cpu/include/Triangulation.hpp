#ifndef TRIANGULATION_H_INCLUDED
#define TRIANGULATION_H_INCLUDED

#include <vector>
#include <Point.hpp>

namespace ced
{
    namespace cpu
    {
        struct triangulation
        {
            // both indexed by half-edge ids
            // half-edges e are the indices into both of delaunators output
            //  -----------------------------------------------------------------------
            /// @build : triangles[e] returns the point id where the half edge starts
            //  -----------------------------------------------------------------------
            std::vector<unsigned int> triangles;
            //  -----------------------------------------------------------------------
            /// @build : halfedges[e] returns the opposite half-edge in the adjacent triangle 
            /// or -1 if there is not adjacent tirangle
            //  -----------------------------------------------------------------------
            std::vector<unsigned int> halfedges; 
            //  -----------------------------------------------------------------------
            /// @build : triangulate 
            /// @param[_in] : list of points
            /// @return[_out] : list of points ordered in [p1, p2, p3] of a triangle;
            //  -----------------------------------------------------------------------
            void triangulate(std::vector<Point>& _points);
        };        
    }
}

#endif // TRIANGULATION_H_INCLUDED
