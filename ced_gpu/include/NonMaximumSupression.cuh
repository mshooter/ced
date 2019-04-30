#ifndef NONMAXIMUMSUPRESSION_CUH_INCLUDED
#define NONMAXIMUMSUPRESSION_CUH_INCLUDED

#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include "ThrustFunctors.cuh"
namespace ced
{
    namespace gpu
    {
        thrust::device_vector<int> d_nonMaximumSupression(    thrust::device_vector<float>& _directions,
                                        thrust::device_vector<float>& _magnitudes, 
                                        float _max_value,
                                        int _height,
                                        int _width, 
                                        int _num_of_pix);
        //  ------------------------------------------------------------------------------------------------------
        // orientation, c, n, ne, nw, s, se, sw, e, w
        struct is_edge
        {
            __host__ __device__ 
            int operator()(const thrust::tuple<float, float, float, float, float, float, float, float, float, float>& t)
            {
                float angle = thrust::get<0>(t);
                float c = thrust::get<1>(t);
                float n = thrust::get<2>(t);
                float ne = thrust::get<3>(t);
                float nw = thrust::get<4>(t);
                float s = thrust::get<5>(t);
                float se = thrust::get<6>(t);
                float sw = thrust::get<7>(t);
                float e = thrust::get<8>(t);
                float w = thrust::get<9>(t);

                if((angle >= 0  && angle <= 22.5f) || (angle >= 157.5f && angle <=180))
                {
                    if(c > w && c > e) return 1; 
                }

                else if(angle >= 22.5f  && angle <= 67.5f)
                {
                    if(c > nw && c > se) return 1; 
                }

                else if(angle >= 67.5f  && angle <= 112.5f)
                {
                    if(c > n && c > s) return 1; 
                }

                else if(angle >= 112.5f  && angle <= 157.5f)
                {
                    if(c > nw && c > se) return 1; 
                }
                else
                {
                    return 0;
                }
            }
        };
        //  ------------------------------------------------------------------------------------------------------
        struct is_Upper
        {
            const float max_value; is_Upper(float _max_value) : max_value(_max_value){}
            __host__ __device__
            int operator()(const float& f) 
            {
                if(f > max_value)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
        };
        //  ------------------------------------------------------------------------------------------------------
        struct isInBound
        {
            const int height;
            const int width;
            isInBound(int _h, int _w) : height(_h), width(_w) {}
            __host__ __device__ 
            int operator()(const int& i)
            {
                int w = i % width;
                int h = i / width;
                if(w >= 1 && h >= 1 && w <( width-1) &&( h < height-1))
                {
                    return 1; 
                }
                else
                {
                    return 0;
                }
            }
        };
       

    }
}


#endif // NONMAXIMUMSUPRESSION_CUH_INCLUDED
