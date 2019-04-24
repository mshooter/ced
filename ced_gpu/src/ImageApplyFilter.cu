#include "ImageApplyFilter.cuh"

namespace ced
{
    namespace gpu
    {
        __global__ void d_applyFilter(  float* _d_oimage_ptr, 
                                        float* _d_nimage_ptr, 
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
                    int base = (j + i * _nwidth) * 3;
                    int ibase = ((w+j) + (i+h) * (_nwidth + _dimension -1)) * 3;
                    int fbase = y;
                    _d_nimage_ptr[base + 0] += _d_oimage_ptr[ibase+0] * _d_filter_ptr[fbase];
                    _d_nimage_ptr[base + 1] += _d_oimage_ptr[ibase+1] * _d_filter_ptr[fbase];
                    _d_nimage_ptr[base + 2] += _d_oimage_ptr[ibase+2] * _d_filter_ptr[fbase];
                }
            }
        }
    }
}
