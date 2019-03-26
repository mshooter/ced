#ifndef CIRCUMCIRCLE_H_INCLUDED
#define CIRCUMCIRCLE_H_INCLUDED

#include <cmath>
#include "Point.hpp"

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
        bool isPointInCircle(Point A, Point B, Point C, Point D); 
        //  -------------------------------------------------------------------------
        /// @build : returns the circum radius
        /// @param[_in] A : first point of the triangle 
        /// @param[_in] B : second point of the triangle 
        /// @param[_in] C : third point of the triangle 
        //  -------------------------------------------------------------------------
        float circumRadius(Point A, Point B, Point C);
        //  -------------------------------------------------------------------------
        /// @build : returns the circum center 
        /// @param[_in] A : first point of the triangle 
        /// @param[_in] B : second point of the triangle 
        /// @param[_in] C : third point of the triangle 
        //  -------------------------------------------------------------------------
        Point circumCenter(Point A, Point B, Point C);
    }
}

#endif // CIRCUMCIRCLE_H_INCLUDED
