#include "Hysterysis.cuh"
#include "ThrustFunctors.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h> 
#include <thrust/transform.h> 

inline int iDivUp(const unsigned int &a, const unsigned int &b){return (a%b != 0 ? (a/b+1) : (a/b));}

namespace ced
{
    namespace gpu
    {
        void hysterysis(    thrust::device_vector<float>& _d_red,
                            thrust::device_vector<float>& _d_green,
                            thrust::device_vector<float>& _d_blue,
                            int _height,
                            int _width, 
                            int _minValue, 
                            int _maxValue)
        {

            
            // upper bound
            thrust::device_vector<float> d_nred(_d_red.size(), 0.0f);
            thrust::device_vector<float> d_ngreen(_d_green.size(), 0.0f);
            thrust::device_vector<float> d_nblue(_d_blue.size(), 0.0f);

            thrust::transform_if(_d_red.begin()     , _d_red.end()   , d_nred.begin()    , set_value(1.0f), isUpper_bound(_maxValue/3.0f));
            thrust::transform_if(_d_green.begin()   , _d_green.end() , d_ngreen.begin()  , set_value(1.0f), isUpper_bound(_maxValue/3.0f));
            thrust::transform_if(_d_blue.begin()    , _d_blue.end()  , d_nblue.begin()   , set_value(1.0f), isUpper_bound(_maxValue/3.0f));

            // thust must work
            // thust lower work
            thrust::transform_if(_d_red.begin()     , _d_red.end()   , d_nred.begin()    , set_value(0.0f), isLower_bound(_minValue/3.0f));
            thrust::transform_if(_d_green.begin()   , _d_green.end() , d_ngreen.begin()  , set_value(0.0f), isLower_bound(_minValue/3.0f));
            thrust::transform_if(_d_blue.begin()    , _d_blue.end()  , d_nblue.begin()   , set_value(0.0f), isLower_bound(_minValue/3.0f));
            // currentPixel check if its smaller than the min value 
            // check if any of the neighbours cells is bigger than the minvalue            
            //  ------------------------ type cast---------------------------------------------
            float* red_ptr      = thrust::raw_pointer_cast(_d_red.data());
            float* green_ptr    = thrust::raw_pointer_cast(_d_green.data());
            float* blue_ptr     = thrust::raw_pointer_cast(_d_blue.data());
            float* nred_ptr     = thrust::raw_pointer_cast(d_nred.data());
            float* ngreen_ptr   = thrust::raw_pointer_cast(d_ngreen.data());
            float* nblue_ptr    = thrust::raw_pointer_cast(d_nblue.data());
            //  ------------------------ init exec---------------------------------------------
            int blockW = 32;       
            int blockH = 32;       
            const dim3 grid(iDivUp(_width, blockW), iDivUp(_height, blockH)); 
            const dim3 threadBlock(blockW, blockH); 
            //  ------------------------ call kernel-------------------------------------------
            d_lowerBoundNeighbours<<<grid, threadBlock>>>(  red_ptr,
                                                            green_ptr,
                                                            blue_ptr,
                                                            nred_ptr,
                                                            ngreen_ptr,
                                                            nblue_ptr,
                                                            _height,
                                                            _width,
                                                            _minValue); 
            cudaDeviceSynchronize();
            thrust::copy(   thrust::make_zip_iterator(  thrust::make_tuple(d_nred.begin(), d_ngreen.begin(), d_nblue.begin())), 
                            thrust::make_zip_iterator(  thrust::make_tuple(d_nred.end(), d_ngreen.end(), d_nblue.end())),
                            thrust::make_zip_iterator(  thrust::make_tuple(_d_red.begin(), _d_green.begin(), _d_blue.begin())));
        }
        //  -------------------------------------------------------------------------------------
        __global__ void d_lowerBoundNeighbours(   float* _redPtr,
                                                  float* _greenPtr,
                                                  float* _bluePtr,
                                                  float* _nredPtr,
                                                  float* _ngreenPtr,
                                                  float* _nbluePtr,
                                                  int _height,
                                                  int _width, 
                                                  int _minValue
                                                  )
        {
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            const int i = blockIdx.y * blockDim.y + threadIdx.y;
            if( (j > 1 && j < (_width-1)) && (
                (i > 1 && i < _height -1)))
            {
                // check if any value is _minvalue 
                const int currentIndex = (j     + i     * _width);
                const int top          = (j     + (i+1) * _width);
                const int bottom       = (j     + (i-1) * _width);
                const int left         = ((j-1) + i     * _width);
                const int topleft      = ((j-1) + (i+1) * _width);
                const int bottomleft   = ((j-1) + (i-1) * _width);
                const int right        = ((j+1) + i     * _width);
                const int topright     = ((j+1) + (i+1) * _width);
                const int bottomright  = ((j+1) + (i-1) * _width);
        
                float topPix          = (_redPtr[ top          ] + _greenPtr[ top          ] + _bluePtr[ top          ])/3.0f;
                float bottomPix       = (_redPtr[ bottom       ] + _greenPtr[ bottom       ] + _bluePtr[ bottom       ])/3.0f;
                float leftPix         = (_redPtr[ left         ] + _greenPtr[ left         ] + _bluePtr[ left         ])/3.0f;
                float topLeftPix      = (_redPtr[ topleft      ] + _greenPtr[ topleft      ] + _bluePtr[ topleft      ])/3.0f;
                float bottomLeftPix   = (_redPtr[ bottomleft   ] + _greenPtr[ bottomleft   ] + _bluePtr[ bottomleft   ])/3.0f;
                float rightPix        = (_redPtr[ right        ] + _greenPtr[ right        ] + _bluePtr[ right        ])/3.0f;
                float topRightPix     = (_redPtr[ topright     ] + _greenPtr[ topright     ] + _bluePtr[ topright     ])/3.0f; 
                float bottomRightPix  = (_redPtr[ bottomright  ] + _greenPtr[ bottomright  ] + _bluePtr[ bottomright  ])/3.0f;
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
                    _nredPtr[currentIndex]   = 1.0f;
                    _ngreenPtr[currentIndex] = 1.0f;
                    _nbluePtr[currentIndex]  = 1.0f;
                }
                 
            }
        } 
    }
}
