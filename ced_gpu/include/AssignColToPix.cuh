#ifndef ASSIGNCOLTOPIX_CUH_INCLUDED
#define ASSIGNCOLTOPIX_CUH_INCLUDED

#include <thrust/device_vector.h>
namespace ced
{
    namespace gpu
    {
        void assignColToPix(    thrust::device_vector<float>& _d_red,
                                thrust::device_vector<float>& _d_green,
                                thrust::device_vector<float>& _d_blue,
                                thrust::device_vector<int>&  _d_triID,
                                thrust::device_vector<int>&  _d_pixID,
                                int& _amountOfTris);
    }
}

#endif// ASSIGNCOLTOPIX_CUH_INCLUDED
