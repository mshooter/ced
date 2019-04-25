#include "Hysterysis.hpp"

namespace ced
{
    namespace cpu
    {
        void hysterysis(    std::vector<float>& _red, 
                            std::vector<float>& _green,
                            std::vector<float>& _blue,
                            int _height, 
                            int _width, 
                            float _minValue, 
                            float _maxValue)
        {
            // kif pixel is highter than upper threshold the pixel is accepted as edge
            // if pixel is lower than lower threshold it is rejected
            // if inbetween - then it will be accepted only if it is connected to a pixel that is above the upper threshold 
           
            std::vector<float> red(_height * _width, 0.0f);
            std::vector<float> green(_height * _width, 0.0f);
            std::vector<float> blue(_height * _width, 0.0f);
            for(int i=1; i < _height-1; ++i)
            {
                for(int j=1; j < _width-1; ++j)
                {
                    const unsigned int currentIndex = (j     + i     * _width);
                    const unsigned int top          = (j     + (i+1) * _width);
                    const unsigned int bottom       = (j     + (i-1) * _width);
                    const unsigned int left         = ((j-1) + i     * _width);
                    const unsigned int topleft      = ((j-1) + (i+1) * _width);
                    const unsigned int bottomleft   = ((j-1) + (i-1) * _width);
                    const unsigned int right        = ((j+1) + i     * _width);
                    const unsigned int topright     = ((j+1) + (i+1) * _width);
                    const unsigned int bottomright  = ((j+1) + (i-1) * _width);
        
                    float currentPix      = (_red[ currentIndex ] + _green[ currentIndex ] + _blue[ currentIndex ])/3.0f;
                    float topPix          = (_red[ top          ] + _green[ top          ] + _blue[ top          ])/3.0f;
                    float bottomPix       = (_red[ bottom       ] + _green[ bottom       ] + _blue[ bottom       ])/3.0f;
                    float leftPix         = (_red[ left         ] + _green[ left         ] + _blue[ left         ])/3.0f;
                    float topLeftPix      = (_red[ topleft      ] + _green[ topleft      ] + _blue[ topleft      ])/3.0f;
                    float bottomLeftPix   = (_red[ bottomleft   ] + _green[ bottomleft   ] + _blue[ bottomleft   ])/3.0f;
                    float rightPix        = (_red[ right        ] + _green[ right        ] + _blue[ right        ])/3.0f;
                    float topRightPix     = (_red[ topright     ] + _green[ topright     ] + _blue[ topright     ])/3.0f;
                    float bottomRightPix  = (_red[ bottomright  ] + _green[ bottomright  ] + _blue[ bottomright  ])/3.0f;
                    
                    if(currentPix < _minValue)
                    {
                        red[currentIndex]   = 0.0f;
                        green[currentIndex] = 0.0f;
                        blue[currentIndex]  = 0.0f;
                    }
                    else if(currentPix > _maxValue)
                    {
                        red[currentIndex]   = 1.0f;
                        green[currentIndex] = 1.0f;
                        blue[currentIndex]  = 1.0f;
                    }
                    else
                    {
                        if(
                                topPix > _minValue ||
                                bottomPix > _minValue ||
                                leftPix > _minValue ||
                                topLeftPix > _minValue ||
                                bottomLeftPix > _minValue ||
                                rightPix > _minValue ||
                                topRightPix > _minValue ||
                                bottomRightPix > _minValue
                        )
                        {
                            red[currentIndex]   = 1.0f;
                            green[currentIndex] = 1.0f;
                            blue[currentIndex]  = 1.0f;
                        }
                    }
                }
            }
            _red = std::move(red);
            _green = std::move(green);
            _blue = std::move(blue);
        }
    }

}
