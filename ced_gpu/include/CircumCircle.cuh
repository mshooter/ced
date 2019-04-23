#ifndef CIRCUMCIRCLE_CUH_INCLUDED
#define CIRCUMCIRCLE_CUH_INCLUDED
    
#include "Point.cuh"
#include <limits>

namespace ced
{
    namespace gpu
    {
        struct circumRadius
        {
            const Point A;
            const Point B;
            circumRadius(Point _p0, Point _p1) : A(_p0), B(_p1) {}
            __host__ __device__
            float operator()(const Point& C)
            {
                float delta_abx = B.x-A.x;
                float delta_aby = B.y-A.y;
                float delta_acx = C.x-A.x;
                float delta_acy = C.y-A.y;
                
                const float dist_ab = delta_abx * delta_abx + delta_aby * delta_aby;
                const float dist_ac = delta_acx * delta_acx + delta_acy * delta_acy;
                const float N = delta_abx * delta_acy - delta_aby * delta_acx;

                const float x = (delta_acy * dist_ab - delta_aby * dist_ac) * 0.5f / N; 
                const float y = (delta_acx * dist_ac - delta_acx * dist_ab) * 0.5f / N; 

                //this is weird ? must check
                if(dist_ab != 0 && dist_ac != 0 && N != 0)
                {
                    return x * x + y * y;
                }
                else
                {
                    return std::numeric_limits<float>::max();
                } 
            }
        };
    }
}

#endif // CIRCUMCIRCLE_CUH_INCLUDED
