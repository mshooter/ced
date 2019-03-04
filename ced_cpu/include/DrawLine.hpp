// @file : DrawLine.hpp
#ifndef DRAWLINE_H_INCLUDED
#define DRAWLINE_H_INCLUDED

#include <vector>
#include "Point.hpp"

// TODO: parameter input should bu the average color for each triangle 

namespace ced
{
    namespace cpu
    {
        //--------------------------------------------------------------
        /// @brief : drawline on image
        /// @param[_in] _p1: starting point
        /// @param[_in] _p2: end point
        /// @param[_in] _image: vector that holds image values
        /// @param[_in] _width: the width of the image
        //--------------------------------------------------------------
        void drawLine(Point _p1, Point _p2, std::vector<float>& _image, int _width);   
    }
}

#endif // DRAWLINE_H_INCLUDED
