#include "NonMaximumSupression.hpp"
#include <iostream>

namespace ced
{
    void nonMaximumSupression(int& _height, int& _width, std::vector<float> _orientation, std::vector<float>& _pixelData)
    {
        int height = _height;
        int width = _width;
        // nimage  = _ pixelData
        std::vector<float> nimage(_width * _height * 3, 0.0f);
        for(int k=0; k < 3; ++k)
        {
            for(int i=1; i < height-1; ++i)
            {
                for(int j = 1; j < width-1; ++j)
                {
                    // iterate over all angles
                    int angle = _orientation[(j)+(i)*_width];
                    // index neighbours 
                    const unsigned int currentIndex = (j     + i     * _width) * 3 + k;
                    const unsigned int top          = (j     + (i+1) * _width) * 3 + k;
                    const unsigned int bottom       = (j     + (i-1) * _width) * 3 + k;
                    const unsigned int left         = ((j-1) + i     * _width) * 3 + k;
                    const unsigned int topleft      = ((j-1) + (i+1) * _width) * 3 + k;
                    const unsigned int bottomleft   = ((j-1) + (i-1) * _width) * 3 + k;
                    const unsigned int right        = ((j+1) + i     * _width) * 3 + k;
                    const unsigned int topright     = ((j+1) + (i+1) * _width) * 3 + k;
                    const unsigned int bottomright  = ((j+1) + (i-1) * _width) * 3 + k;
                    // if maxium and magnitude > upper threshold = pixel is an edge
                    // yes then keep it, else turn it black
                    float currentPix      = _pixelData[ currentIndex ];
                    float topPix          = _pixelData[ top          ];
                    float bottomPix       = _pixelData[ bottom       ];
                    float leftPix         = _pixelData[ left         ];
                    float topLeftPix      = _pixelData[ topleft      ];
                    float bottomLeftPix   = _pixelData[ bottomleft   ];
                    float rightPix        = _pixelData[ right        ];
                    float topRightPix     = _pixelData[ topright     ];
                    float bottomRightPix  = _pixelData[ bottomright  ];
                    
                    switch(angle)
                     {
                         case 0:
                             if(currentPix < leftPix || currentPix < rightPix)   { nimage[currentIndex] = 0.0f;}
                             else  {nimage[currentIndex] = currentPix;}
                             break;
                         case 45:
                             if(currentPix < bottomRightPix || currentPix < topLeftPix)  {nimage[currentIndex] = 0.0f;}
                             else  {nimage[currentIndex] = currentPix;}
                             break;
                         case 90:
                             if(currentPix < topPix || currentPix < bottomPix)  {nimage[currentIndex] = 0.0f;}
                             else  {nimage[currentIndex] = currentPix;}
                             break;
                         case 135:
                             if(currentPix < bottomLeftPix || currentPix < topRightPix)  {nimage[currentIndex] = 0.0f;}
                             else  {nimage[currentIndex] = currentPix;}
                             break;
                         default:
                             nimage[currentIndex] = currentPix;
                             break;
                     }
                }
            } 
        }
       _pixelData = std::move(nimage);
    }
}
