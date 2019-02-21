#include "Hysterysis.hpp"

namespace ced
{
    void hysterysis(std::vector<float>& _nonMax, int _height, int _width, float _minValue, float _maxValue)
    {
        // kif pixel is highter than upper threshold the pixel is accepted as edge
        // if pixel is lower than lower threshold it is rejected
        // if inbetween - then it will be accepted only if it is connected to a pixel that is above the upper threshold 
       
        std::vector<float> nimage(_height * _width * 3, 0.0f);
        for(int k=0; k < 3; ++k)
        {
            for(int i=1; i < _height-1; ++i)
            {
                for(int j=1; j < _width-1; ++j)
                {
                    const unsigned int currentIndex = (j     + i     * _width) * 3 + k;
                    const unsigned int top          = (j     + (i+1) * _width) * 3 + k;
                    const unsigned int bottom       = (j     + (i-1) * _width) * 3 + k;
                    const unsigned int left         = ((j-1) + i     * _width) * 3 + k;
                    const unsigned int topleft      = ((j-1) + (i+1) * _width) * 3 + k;
                    const unsigned int bottomleft   = ((j-1) + (i-1) * _width) * 3 + k;
                    const unsigned int right        = ((j+1) + i     * _width) * 3 + k;
                    const unsigned int topright     = ((j+1) + (i+1) * _width) * 3 + k;
                    const unsigned int bottomright  = ((j+1) + (i-1) * _width) * 3 + k;

                    float currentPix      = _nonMax[ currentIndex ];
                    float topPix          = _nonMax[ top          ];
                    float bottomPix       = _nonMax[ bottom       ];
                    float leftPix         = _nonMax[ left         ];
                    float topLeftPix      = _nonMax[ topleft      ];
                    float bottomLeftPix   = _nonMax[ bottomleft   ];
                    float rightPix        = _nonMax[ right        ];
                    float topRightPix     = _nonMax[ topright     ];
                    float bottomRightPix  = _nonMax[ bottomright  ];
                    
                    if(currentPix < _minValue) {nimage[currentIndex] = 0.0f;}
                    else if(currentPix > _maxValue)  {nimage[currentIndex] = 1.0f;}
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
                            nimage[currentIndex] = 1.0f;
                            nimage[currentIndex] = 1.0f;
                            nimage[currentIndex] = 1.0f;
                        }
                    }
                }
            }
        }
        _nonMax = std::move(nimage);
    }
}
