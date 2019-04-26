#include "AssignAvgColKernel.cuh"

namespace ced
{
    namespace gpu
    {
        __global__ void assignAvgCol(   float* _d_oredPtr,
                                        float* _d_ogreenPtr,
                                        float* _d_obluePtr,
                                        float* _d_nredPtr,
                                        float* _d_ngreenPtr,
                                        float* _d_nbluePtr,
                                        int*   _d_pixIDptr,
                                        int    _sizeOfPixIDs)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x; 
            if(i < _sizeOfPixIDs)
            {
                int id = _d_pixIDptr[i];
                _d_nredPtr[id]  = _d_oredPtr[id];
                _d_ngreenPtr[id]= _d_ogreenPtr[id];
                _d_nbluePtr[id] = _d_obluePtr[id];
            }
        }
    }
}
