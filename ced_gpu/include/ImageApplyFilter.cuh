#ifndef IMAGEAPPLYFILTER_CUH_INCLUDED
#define IMAGEAPPLYFILTER_CUH_INCLUDED

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
                                        int _dimension);
    }
}

#endif // IMAGEAPPLYFILTER_CUH_INCLUDED
