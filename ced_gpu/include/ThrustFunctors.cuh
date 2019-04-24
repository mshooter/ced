#ifndef THRUSTFUNCTORS_CUH_INCLUDED
#define THRUSTFUNCTORS_CUH_INCLUDED

#include <thrust/tuple.h>

namespace ced
{
    namespace gpu
    {
        // TODO: might move this function to the create triangle file
        //  ------------------------------------------------------------------------------------------ 
        // @build : get the minimum if
        //  ------------------------------------------------------------------------------------------ 
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
        // @build : multiply by constant
        //  ------------------------------------------------------------------------------------------ 
        struct multiplyConst
        {
            const float a; 
            multiplyConst(float _a) : a(_a) {}
            __host__ __device__ 
            float operator()(const float& id)
            {
                return id / a;
            }
    
        };
    }
}

#endif // THRUSTFUNCTORS_CUH_INCLUDED
