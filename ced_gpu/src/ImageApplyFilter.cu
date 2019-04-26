#include "ImageApplyFilter.cuh"

namespace ced
{
    namespace gpu
    {
        __global__ void d_applyFilter(  float* _d_ored_ptr, 
                                        float* _d_ogreen_ptr,
                                        float* _d_oblue_ptr,
                                        float* _d_nred_ptr, 
                                        float* _d_ngreen_ptr, 
                                        float* _d_nblue_ptr, 
                                        float* _d_filter_ptr,
                                        int _nwidth,
                                        int _nheight, 
                                        int _dimension)
        {
            // pixel coordinates
            const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; 
            const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
            if(j < _nwidth && i < _nheight)
            {
                for(int y = 0; y < _dimension * _dimension; ++y)
                {
                    int w = y % _dimension; 
                    int h = y / _dimension;
                    int base = (j + i * _nwidth);
                    int ibase = ((w+j) + (i+h) * (_nwidth + _dimension -1));
                    int fbase = y;
                    _d_nred_ptr[base]   += _d_ored_ptr[ibase] * _d_filter_ptr[fbase];
                    _d_ngreen_ptr[base] += _d_ogreen_ptr[ibase] * _d_filter_ptr[fbase];
                    _d_nblue_ptr[base]  += _d_oblue_ptr[ibase] * _d_filter_ptr[fbase];
                }
            }
        }
    }
}
