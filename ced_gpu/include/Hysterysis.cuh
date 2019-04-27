#ifndef HYSTERYSIS_CUH_INCLUDED
#define HYSTERYSIS_CUH_INCLUDED

#include <thrust/device_vector.h>

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
                            int _maxValue);

        __global__ void d_lowerBoundNeighbours(   float* _redPtr,
                                                  float* _greenPtr,
                                                  float* _bluePtr,
                                                  float* _nredPtr,
                                                  float* _ngreenPtr,
                                                  float* _nbluePtr,
                                                  int _height,
                                                  int _width, 
                                                  int _minValue
                                                  );

    }
}

#endif // HYSTERYSIS_CUH_INCLUDED
