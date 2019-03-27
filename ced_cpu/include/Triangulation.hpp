#ifndef TRIANGULATION_H_INCLUDED
#define TRIANGULATION_H_INCLUDED

#include <vector>
#include <Point.hpp>

// TODO: Need to implement the link function and add triangle
// TODO: and do not have a class triangulation, have a function!
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
        void addTriangle(T _id0, T _id1, T _id2, T _ahe0, T _ahe1, T _ahe2, std::vector<T>& _triangles);
        //  -----------------------------------------------------------------------
        /// @build : triangulate 
        /// @param[_in] : list of points
        /// @return[_out] : list of points ordered in [p1, p2, p3] of a triangle;
        //  -----------------------------------------------------------------------
        void triangulate(std::vector<Point>& _points);
        //  -----------------------------------------------------------------------
        /// @build: triangulation data structure
        //  -----------------------------------------------------------------------
        struct triangulation
        {
            //  -----------------------------------------------------------------------
            /// @build : triangles[e] returns the point id where the half edge starts
            //  -----------------------------------------------------------------------
            std::vector<unsigned int> triangles;
            //  -----------------------------------------------------------------------
            /// @build : halfedges[e] returns the opposite half-edge in the adjacent triangle 
            /// or -1 if there is not adjacent tirangle
            //  -----------------------------------------------------------------------
            std::vector<unsigned int> halfedges; 

            std::vector<unsigned int> hull_e_prev; // edge to the previous edge 
            std::vector<unsigned int> hull_e_next; // edge to the next edge 
            std::vector<unsigned int> hull_e_triAdj; // edge to adjacent triangle   
            std::vector<unsigned int> hash; // angular edge hash   
            unsigned int hash_size; // angular edge hash   
            unsigned int hull_start;

        };        
    }
}

#endif // TRIANGULATION_H_INCLUDED
