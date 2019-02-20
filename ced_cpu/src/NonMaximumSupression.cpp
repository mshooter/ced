#include "NonMaximumSupression.hpp"

namespace ced
{
    void nonMaximumSupression()
    {
        // iterate over all pixels
        for(int i=0; i < _height; ++i)
        {
            for(int j =0; j < _width; ++j)
            {
                // iterate over all angles
                float angle = _orientation[j+i*_width];
                // get pixel value
                float rV = _pixelData[(j+i*_width) * 3 + 0];
                float gV = _pixelData[(j+i*_width) * 3 + 1];
                float bV = _pixelData[(j+i*_width) * 3 + 2];
                // if maxium and magnitude > upper threshold = pixel is an edge
                // compare current gradient magnitude with other pixels, is it max ? 
                if(angle >= 22.5f && angle <= 67.5f) // top left pixel & bottom right pixel 
                if(angle >= 67.5f && angle <= 112.5f) // pixel top and below
                if(angle >= 112.5f && angle <= 157.5f) // top right pixel & bottom left 
                if((angle >= 0 && angle <= 22.5f) || (157.5f >= angle && 180 <= angle)) // left pixel and right
                else    // push back value zero into image;
            }
        } 
    }
}
