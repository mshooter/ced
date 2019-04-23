#ifndef THRUSTFUNCTORS_CUH_INCLUDED
#define THRUSTFUNCTORS_CUH_INCLUDED

#include <thrust/tuple.h>

namespace ced
{
    namespace gpu
    {
        struct min_if
        {
            const int i0; 
            const int i1; 
            min_if(int _i0, int _i1) : i0(_i0), i1(_i1){}
            __host__ __device__
            bool operator()(const thrust::tuple<float, int>& curr, const thrust::tuple<float, int>& rhs)
            {
                if((curr.get<1>() != i0) && (curr.get<1>() != i1))
                {   
                    return curr.get<0>() < rhs.get<0>();
                }
                else
                {
                    return false;
                }
            }
        };
        //  ------------------------------------------------------------------------------------------ 
    }
}

#endif // THRUSTFUNCTORS_CUH_INCLUDED
