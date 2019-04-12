#ifndef GETWHITEPIXELS_H_INCLUDED
#define GETWHITEPIXELS_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        //  -----------------------------------------------------------
        /// @build : get the edge pixels
        /// @param[_in] : pixelCoordinates
        /// @param[_in] : imgPix is the data of the image
        /// @param[_in] : the height and the width of the image
        //  -----------------------------------------------------------
        void getWhitePixelsCoords(  std::vector<Point>& _whitePixelsCoord,
                                    std::vector<Point>& _allPixCoord,
                                    std::vector<float> _imgPix,
                                    int height, 
                                    int width);
    }
}

#endif // GETWHITEPIXELS_H_INCLUDED
