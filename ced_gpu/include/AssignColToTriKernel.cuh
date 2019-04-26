#ifndef ASSIGNCOLTOTRI_CUH_INCLUDED
#define ASSIGNCOLTOTRI_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        __global__ void d_assignColToTriKernel(   float* _d_redPtr,
                                                  float* _d_greenPtr,
                                                  float* _d_bluePtr,
                                                  int* _d_pixIDsPtr,
                                                  float _r, 
                                                  float _g,
                                                  float _b,
                                                  int sizeOfPixIDs 
                                              );
    }
}

#endif // ASSIGNCOLTOTRI_CUH_INCLUDED
