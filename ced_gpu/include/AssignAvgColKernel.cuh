#ifndef ASSIGNAVGCOL_CUH_INCLUDED
#define ASSIGNAVGCOL_CUH_INCLUDED

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
                                        int    _sizeOfPixIDs);
    }
}

#endif //ASSIGNAVGCOL_CUH_INCLUDED
