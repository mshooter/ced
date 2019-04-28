#ifndef MATH_CUH_INCLUDED
#define MATH_CUH_INCLUDED

#include <thrust/tuple.h>
namespace ced
{
    namespace gpu
    {
        // --------------------------------------------------------
        // @build : divide by constant gpu
        // @param[_in] : a constant a 
        // --------------------------------------------------------
        template <typename T>
        struct divideByConstant
        {
            const T a; 
            divideByConstant(T _a) : a(_a) {}
            __device__
            float operator()(const float& element)
            {
                return element / a;
            } 
        }; 

        // --------------------------------------------------------
        // @build : add three vectors
        // --------------------------------------------------------
        struct add_three_vectors
        {
            __host__ __device__ 
            float operator()(const thrust::tuple<float, float, float>& t)
            {
                return thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t);
            }
        };
    }
}

#endif // MATH_CUH_INCLUDED
