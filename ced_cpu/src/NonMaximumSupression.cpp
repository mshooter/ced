#include "NonMaximumSupression.hpp"
#include <iostream>

namespace ced
{
    void nonMaximumSupression(int& _height, int& _width, std::vector<float> _orientation, std::vector<float>& _pixelData)
    {
        int height = _height -1;
        int width = _width -1;

        // iterate over all pixels
        std::vector<float> nimage(height * width * 3, 0.0f);
        for(int i=1; i < height; ++i)
        {
            for(int j =1; j < width; ++j)
            {
                // iterate over all angles
                float angle = _orientation[j+i*_width];
                // index neighbours 
                const unsigned int currentIndex = j     + i     * _width * 3;
                const unsigned int top          = j     + (i+1) * _width * 3;
                const unsigned int bottom       = j     + (i-1) * _width * 3;
                const unsigned int left         = (j-1) + i     * _width * 3;
                const unsigned int topleft      = (j-1) + (i+1) * _width * 3;
                const unsigned int bottomleft   = (j-1) + (i-1) * _width * 3;
                const unsigned int right        = (j+1) + i     * _width * 3;
                const unsigned int topright     = (j+1) + (i+1) * _width * 3;
                const unsigned int bottomright  = (j+1) + (i-1) * _width * 3;
                // if maxium and magnitude > upper threshold = pixel is an edge
                // yes then keep it, else turn it black
                for(unsigned int d = 0; d < 3; ++i)
                {
                    float currentPix      = _pixelData[currentIndex + d];
                    float topPix          = _pixelData[top + d];
                    float bottomPix       = _pixelData[bottom + d];
                    float leftPix         = _pixelData[left + d];
                    float topLeftPix      = _pixelData[topleft + d];
                    float bottomLeftPix   = _pixelData[bottomleft + d];
                    float rightPix        = _pixelData[right + d];
                    float topRightPix     = _pixelData[topright + d];
                    float bottomRightPix  = _pixelData[bottomright + d];
                    
                    switch(angle)
                     {
                         case 0:
                             if(currentPix <= leftPix || currentPix <= rightPix)   { nimage[currentIndex + d] = 0.0f;}
                             else  {nimage[currentIndex + d] = currentPix;}
                             break;
                         case 45:
                             if(currentPix <= bottomRightPix || currentPix <= topLeftPix)  {nimage[currentIndex + d] = 0.0f;}
                             else  {nimage[currentIndex + d] = currentPix;}
                             break;
                         case 90:
                             if(currentPix <= topPix || currentPix <= bottomPix)  {nimage[currentIndex + d] = 0.0f;}
                             else  {nimage[currentIndex + d] = currentPix;}
                             break;
                         case 135:
                             if(currentPix <= bottomLeftPix || currentPix <= topRightPix)  {nimage[currentIndex + d] = 0.0f;}
                             else  {nimage[currentIndex + d] = currentPix;}
                             break;
                         default:
                             else  {nimage[currentIndex + d] = currentPix;}
                             break;
                     }
                }
            }
        } 
        _pixelData.resize(height*width*3);
        _pixelData = std::move(nimage);
        _height = std::move(height);
        _width = std::move(width);
    }
}
