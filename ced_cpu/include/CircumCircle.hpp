#ifndef CIRCUMCIRCLE_H_INCLUDED
#define CIRCUMCIRCLE_H_INCLUDED

#include <cmath>
#include <cstdlib>

namespace ced
{
    namespace cpu
    {
        //  -------------------------------------------------------------------------
        /// @build : checks if the point is in the circumcircle of a certain triangle 
        /// with point A, B, C 
        /// @param[_in] A : first point of the triangle 
        /// @param[_in] B : second point of the triangle 
        /// @param[_in] C : third point of the triangle 
        //  -------------------------------------------------------------------------
        template <typename T>
        bool isPointInCircle(T A, T B, T C, T D); 
        //  -------------------------------------------------------------------------
        /// @build : returns the circum radius
        /// @param[_in] A : first point of the triangle 
        /// @param[_in] B : second point of the triangle 
        /// @param[_in] C : third point of the triangle 
        //  -------------------------------------------------------------------------
        template <typename T, typename U>
        U circumRadius(T A, T B, T C);
        //  -------------------------------------------------------------------------
        /// @build : returns the circum center 
        /// @param[_in] A : first point of the triangle 
        /// @param[_in] B : second point of the triangle 
        /// @param[_in] C : third point of the triangle 
        //  -------------------------------------------------------------------------
        //www.geeksforgeeks.org/program-find-circumcenter-triangle-2/
        template <typename T>
        T circumCenter(T A, T B, T C);
        //  -------------------------------------------------------------------------
        /// @build : template implementations 
        //  -------------------------------------------------------------------------
        #include "CircumCircle.inl"
    }
}

#endif // CIRCUMCIRCLE_H_INCLUDED
