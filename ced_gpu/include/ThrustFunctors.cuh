#ifndef THRUSTFUNCTORS_CUH_INCLUDED
#define THRUSTFUNCTORS_CUH_INCLUDED

#include <thrust/tuple.h>

namespace ced
{
    namespace gpu
    {
        struct is_identity
        {
            __host__ __device__
            bool operator()(const int& t)
            {
                return t == 1;
            }
        };
        //  ------------------------------------------------------------------------------------------------------
        struct find_neighbours 
        {
            const int width;
            find_neighbours(int _w) : width(_w) {}
            __host__ __device__ 
            thrust::tuple<int, int, int, int, int, int, int, int> operator()(const int& i)
            {
                thrust::tuple<int, int, int, int, int, int, int, int> result;
                thrust::get<0>(result) = i - width;
                thrust::get<1>(result) = i +1 - width;
                thrust::get<2>(result) = i - 1 - width;
                thrust::get<3>(result) = i + width;
                thrust::get<4>(result) = i + (width+1);
                thrust::get<5>(result) = i + (width-1);
                thrust::get<6>(result) = i + 1;
                thrust::get<7>(result) = i - 1;
                return result;
            }
        };
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
        struct divideZip
        {
            __host__ __device__
            thrust::tuple<float, float, float> operator()(const thrust::tuple<float, float, float>& t, const float& d)
            {
                thrust::tuple <float, float, float> result;
                thrust::get<0>(result) = thrust::get<0>(t)/ d;
                thrust::get<1>(result) = thrust::get<1>(t)/ d;
                thrust::get<2>(result) = thrust::get<2>(t)/ d;
                return result;
            }
        };
        //  ----------------------------------------------------------------------------------------
        struct isLower_bound
        {
            const float min_value;
            isLower_bound(float _min_value) : min_value(_min_value){}
            __host__ __device__
            bool operator()(const float& t)
            {
                if(t < min_value)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        };
        //  ----------------------------------------------------------------------------------------
        struct isUpper_bound
        {
            const float max_value;
            isUpper_bound(float _max_value) : max_value(_max_value){}
            __host__ __device__
            bool operator()(const float& t)
            {
                if(t > max_value)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        };
        //  ----------------------------------------------------------------------------------------
        struct set_value
        {
            const int value;
            set_value(int _value) : value(_value){}
            __host__ __device__
            float operator()(const float& t)
            {
                return value;
            }
        };
        //  ----------------------------------------------------------------------------------------

    }
}

#endif // THRUSTFUNCTORS_CUH_INCLUDED
