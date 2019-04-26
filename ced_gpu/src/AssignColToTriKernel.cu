#include "AssignColToTriKernel.cuh"

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
                                                  int _sizeOfPixIDs 
                                              )
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x; 

            if(i < _sizeOfPixIDs)
            {   
                int id = _d_pixIDsPtr[i];
                _d_redPtr[id]   = _r;
                _d_greenPtr[id] = _g;
                _d_bluePtr[id]  = _b;
            }
        }
    }
}
