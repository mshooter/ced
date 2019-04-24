#ifndef IMAGEAPPLYFILTER_CUH_INCLUDED
#define IMAGEAPPLYFILTER_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        // passing by reference can not be used in kernels
        __global__ void d_applyFilter(  float* _d_oimage_ptr, 
                                        float* _d_nimage_ptr, 
                                        float* _d_filter_ptr,
                                        int _nwidth,
                                        int _nheight, 
                                        int _dimension);
    }
}

#endif // IMAGEAPPLYFILTER_CUH_INCLUDED
