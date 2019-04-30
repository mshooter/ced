#ifndef HYSTERYSIS_CUH_INCLUDED
#define HYSTERYSIS_CUH_INCLUDED

#include <thrust/device_vector.h>
#include "ThrustFunctors.cuh"
namespace ced
{
    namespace gpu
    {
        thrust::device_vector<int> hysterysis(    thrust::device_vector<int>& _edge_ids,
                            thrust::device_vector<float>& _magnitude,
                            float _min_value,
                            float _max_value,
                            int _width
                        );
        //  ---------------------------------------------------------------------------
        struct is_minima
        {
            const float minima;
            is_minima(float _min) : minima(_min){}
            __host__ __device__
            bool operator()(const float& f)
            {
                return f <= minima;
            }
        };
        //  ---------------------------------------------------------------------------
        // c, n, ne, nw, s, se, sw, e, w
        struct is_thinEdge
        {
            const float minima;
            const float max;
            is_thinEdge(float _min, float _max) : minima(_min), max(_max){}
            __host__ __device__
            int operator()(const thrust::tuple<float, float, float, float, float, float, float, float, float>& f)
            {
                float c     = thrust::get<0>(f);
                float n     = thrust::get<1>(f);
                float ne    = thrust::get<2>(f);

                float nw    = thrust::get<3>(f);
                float s     = thrust::get<4>(f);
                float se    = thrust::get<5>(f);

                float sw    = thrust::get<6>(f);
                float e     = thrust::get<7>(f);
                float w     = thrust::get<8>(f);
                if(c < minima) 
                {
                    return 0;
                }
                else if(c > max)
                {
                    return 1;
                }
                else
                {
                    if( n > minima ||
                        s > minima ||
                        w > minima ||
                        nw > minima ||
                        sw > minima ||
                        e > minima ||
                        se > minima ||
                        ne > minima)
                    {
                        return 1;
                    }
                }
            }
        };
    }
}

#endif // HYSTERYSIS_CUH_INCLUDED
