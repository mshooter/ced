#ifndef NONMAXIMUMSUPRESSION_CUH_INCLUDED
#define NONMAXIMUMSUPRESSION_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
         // outside the loop
        __global__ void d_nonMaximumSupression( float* _d_redPtr,
                                                float* _d_greenPtr,
                                                float* _d_bluePtr,
                                                float* _d_nredPtr,
                                                float* _d_ngreenPtr,
                                                float* _d_nbluePtr,
                                                int _height,
                                                int _width);
    }
}


#endif // NONMAXIMUMSUPRESSION_CUH_INCLUDED
