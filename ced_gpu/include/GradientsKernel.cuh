#ifndef GRADIENTSKERNEL_CUH_INCLUDED
#define GRADIENTSKERNEL_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        __global__ void d_calculateGradients(   float* _d_Gx_ptr,
                                                float* _d_pixelData_ptr,
                                                const float* _d_kernelX_ptr,
                                                int _nwidth,
                                                int _nheight);
    }
}

#endif // GRADIENTSKERNEL_CUH_INCLUDED
