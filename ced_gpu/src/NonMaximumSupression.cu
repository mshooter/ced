#include "NonMaximumSupression.cuh"

namespace ced
{
    namespace gpu
    {
        __global__ void d_nonMaximumSupression( float* _d_redPtr,
                                                float* _d_greenPtr,
                                                float* _d_bluePtr,
                                                float* _d_nredPtr,
                                                float* _d_ngreenPtr,
                                                float* _d_nbluePtr,
                                                float* _orientation,
                                                int _height,
                                                int _width)
        {
            const int j = blockIdx.x * blockDim.x + threadIdx.x; 
            const int i = blockIdx.y * blockDim.y + threadIdx.y; 
            if( (j > 1 && j < (_width-1)) && (
                (i > 1 && i < _height -1)))
            {
                const int currentIndex = (j     + i     * _width);
                const int top          = (j     + (i+1) * _width);
                const int bottom       = (j     + (i-1) * _width);
                const int left         = ((j-1) + i     * _width);
                const int topleft      = ((j-1) + (i+1) * _width);
                const int bottomleft   = ((j-1) + (i-1) * _width);
                const int right        = ((j+1) + i     * _width);
                const int topright     = ((j+1) + (i+1) * _width);
                const int bottomright  = ((j+1) + (i-1) * _width);
                // if maxium and magnitude > upper threshold = pixel is an edge
                // yes then keep it, else turn it black
                float currentPix      =(_d_redPtr[ currentIndex ] + _d_greenPtr[currentIndex] + _d_bluePtr[currentIndex])/3.0f  ;
                float topPix          =(_d_redPtr[ top          ] + _d_greenPtr[ top ]        + _d_bluePtr[ top]        )/3.0f  ;
                float bottomPix       =(_d_redPtr[ bottom       ] + _d_greenPtr[ bottom ]     + _d_bluePtr[ bottom]     )/3.0f  ;
                float leftPix         =(_d_redPtr[ left         ] + _d_greenPtr[ left ]       + _d_bluePtr[ left]       )/3.0f  ;
                float topLeftPix      =(_d_redPtr[ topleft      ] + _d_greenPtr[ topleft ]    + _d_bluePtr[ topleft]    )/3.0f  ;
                float bottomLeftPix   =(_d_redPtr[ bottomleft   ] + _d_greenPtr[ bottomleft ] + _d_bluePtr[ bottomleft] )/3.0f  ;
                float rightPix        =(_d_redPtr[ right        ] + _d_greenPtr[ right ]      + _d_bluePtr[ right]      )/3.0f  ;
                float topRightPix     =(_d_redPtr[ topright     ] + _d_greenPtr[ topright ]   + _d_bluePtr[ topright]   )/3.0f  ;
                float bottomRightPix  =(_d_redPtr[ bottomright  ] + _d_greenPtr[ bottomright ]+ _d_bluePtr[ bottomright])/3.0f  ; 
                
                int angle = _orientation[(j)+(i)*_width]; 
                if(angle == 0 && (currentPix < leftPix  || currentPix < rightPix))
                {
                    _d_nredPtr[currentIndex] = 0.0f;
                    _d_ngreenPtr[currentIndex] = 0.0f;
                    _d_nbluePtr[currentIndex] = 0.0f;
                }
                else
                {
                    _d_nredPtr[currentIndex]   = currentPix;
                    _d_ngreenPtr[currentIndex] = currentPix;
                    _d_nbluePtr[currentIndex]  = currentPix;
                }
                if(angle == 45 && (currentPix < bottomRightPix  || currentPix < topLeftPix))
                {
                    _d_nredPtr[currentIndex] = 0.0f;
                    _d_ngreenPtr[currentIndex] = 0.0f;
                    _d_nbluePtr[currentIndex] = 0.0f;
                }
                else
                {
                    _d_nredPtr[currentIndex]   = currentPix;
                    _d_ngreenPtr[currentIndex] = currentPix;
                    _d_nbluePtr[currentIndex]  = currentPix;
                }
                if(angle == 90 && (currentPix < topPix || currentPix < bottomPix))
                { 
                    _d_nredPtr[currentIndex] = 0.0f;
                    _d_ngreenPtr[currentIndex] = 0.0f;
                    _d_nbluePtr[currentIndex] = 0.0f;
                }
                else  
                {
                    _d_nredPtr[currentIndex]   = currentPix;
                    _d_ngreenPtr[currentIndex] = currentPix;
                    _d_nbluePtr[currentIndex]  = currentPix;
                }
                if(angle == 135 && (bottomLeftPix < topPix || currentPix < topRightPix))
                { 
                    _d_nredPtr[currentIndex] = 0.0f;
                    _d_ngreenPtr[currentIndex] = 0.0f;
                    _d_nbluePtr[currentIndex] = 0.0f;
                }
                else  
                {
                    _d_nredPtr[currentIndex]   = currentPix;
                    _d_ngreenPtr[currentIndex] = currentPix;
                    _d_nbluePtr[currentIndex]  = currentPix;
                }
            }
        }
    }
}
