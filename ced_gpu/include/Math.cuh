#ifndef MATH_CUH_INCLUDED
#define MATH_CUH_INCLUDED

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
            template <typename T>
            __device__ 
            void operator()(T t)
            {
                thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<2>(t);
            }
        };  
    }
}

#endif // MATH_CUH_INCLUDED
