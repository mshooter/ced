#include "NonMaximumSupression.hpp"
#include <iostream>

namespace ced
{
    namespace cpu
    {
        void nonMaximumSupression(  int& _height, 
                                    int& _width, 
                                    std::vector<float> _orientation, 
                                    std::vector<float>& _red,
                                    std::vector<float>& _green,
                                    std::vector<float>& _blue)
        {
            int height = _height;
            int width = _width;
            // nimage  = _ pixelData
            std::vector<float> red(_width * _height, 0.0f);
            std::vector<float> green(_width * _height, 0.0f);
            std::vector<float> blue(_width * _height, 0.0f);
            for(int i=1; i < height-1; ++i)
            {
                for(int j = 1; j < width-1; ++j)
                {
                    // iterate over all angles
                    int angle = _orientation[(j)+(i)*_width];
                    // index neighbours 
                    const unsigned int currentIndex = (j     + i     * _width);
                    const unsigned int top          = (j     + (i+1) * _width);
                    const unsigned int bottom       = (j     + (i-1) * _width);
                    const unsigned int left         = ((j-1) + i     * _width);
                    const unsigned int topleft      = ((j-1) + (i+1) * _width);
                    const unsigned int bottomleft   = ((j-1) + (i-1) * _width);
                    const unsigned int right        = ((j+1) + i     * _width);
                    const unsigned int topright     = ((j+1) + (i+1) * _width);
                    const unsigned int bottomright  = ((j+1) + (i-1) * _width);
                    // if maxium and magnitude > upper threshold = pixel is an edge
                    // yes then keep it, else turn it black
                    float currentPix      =(_red[ currentIndex ] + _green[currentIndex] + _blue[currentIndex])/3.0f  ;
                    float topPix          =(_red[ top          ] + _green[ top ]        + _blue[ top]        )/3.0f  ;
                    float bottomPix       =(_red[ bottom       ] + _green[ bottom ]     + _blue[ bottom]     )/3.0f  ;
                    float leftPix         =(_red[ left         ] + _green[ left ]       + _blue[ left]       )/3.0f  ;
                    float topLeftPix      =(_red[ topleft      ] + _green[ topleft ]    + _blue[ topleft]    )/3.0f  ;
                    float bottomLeftPix   =(_red[ bottomleft   ] + _green[ bottomleft ] + _blue[ bottomleft] )/3.0f  ;
                    float rightPix        =(_red[ right        ] + _green[ right ]      + _blue[ right]      )/3.0f  ;
                    float topRightPix     =(_red[ topright     ] + _green[ topright ]   + _blue[ topright]   )/3.0f  ;
                    float bottomRightPix  =(_red[ bottomright  ] + _green[ bottomright ]+ _blue[ bottomright])/3.0f  ;
                    
                    switch(angle)
                     {
                         case 0:
                            if(currentPix < leftPix || currentPix < rightPix)
                            { 
                                red[currentIndex] = 0.0f;
                                green[currentIndex] = 0.0f;
                                blue[currentIndex] = 0.0f;
                            }
                            else
                            {
                                red[currentIndex]   = currentPix;
                                green[currentIndex] = currentPix;
                                blue[currentIndex]  = currentPix;
                            }
                            break;
                         case 45:
                            if(currentPix < bottomRightPix || currentPix < topLeftPix)
                            { 
                                red[currentIndex] = 0.0f;
                                green[currentIndex] = 0.0f;
                                blue[currentIndex] = 0.0f;
                            }
                            else
                            {
                                red[currentIndex]   = currentPix;
                                green[currentIndex] = currentPix;
                                blue[currentIndex]  = currentPix;
                            }
                             break;
                         case 90:
                             if(currentPix < topPix || currentPix < bottomPix)
                             { 
                                 red[currentIndex] = 0.0f;
                                 green[currentIndex] = 0.0f;
                                 blue[currentIndex] = 0.0f;
                             }
                             else  
                             {
                                 red[currentIndex]   = currentPix;
                                 green[currentIndex] = currentPix;
                                 blue[currentIndex]  = currentPix;
                             }
                             break;
                         case 135:
                             if(currentPix < bottomLeftPix || currentPix < topRightPix)
                             { 
                                 red[currentIndex] = 0.0f;
                                 green[currentIndex] = 0.0f;
                                 blue[currentIndex] = 0.0f;
                             }
                             else  
                             {
                                 red[currentIndex]   = currentPix;
                                 green[currentIndex] = currentPix;
                                 blue[currentIndex]  = currentPix;
                             }
                             break;
                         default:
                             {
                                 red[currentIndex]   = currentPix;
                                 green[currentIndex] = currentPix;
                                 blue[currentIndex]  = currentPix;
                             }
                             break;
                     }
                }
            }
           _red = std::move(red);
           _green = std::move(green);
           _blue = std::move(blue);
        }
    }
}
