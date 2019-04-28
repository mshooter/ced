#ifndef CIRCUMCIRCLE_CUH_INCLUDED
#define CIRCUMCIRCLE_CUH_INCLUDED
    
#include <thrust/tuple.h>
namespace ced
{
    namespace gpu
    {
        struct circumRadius
        {
            const float Ax;
            const float Ay;
            const float Bx;
            const float By;
            circumRadius(   float _p0x, 
                            float _p0y, 
                            float _p1x, 
                            float _p1y) : 
                            Ax(_p0x), 
                            Ay(_p0y), 
                            Bx(_p1x), 
                            By(_p1y) {}
            __host__ __device__
            float operator()(const thrust::tuple<float, float>& C)
            {
                float delta_abx = Bx-Ax;
                float delta_aby = By-Ay;
                float delta_acx = C.get<0>()-Ax;
                float delta_acy = C.get<1>()-Ay;
                
                const float dist_ab = delta_abx * delta_abx + delta_aby * delta_aby;
                const float dist_ac = delta_acx * delta_acx + delta_acy * delta_acy;
                const float N = delta_abx * delta_acy - delta_aby * delta_acx;

                const float x = (delta_acy * dist_ab - delta_aby * dist_ac) * 0.5f / N; 
                const float y = (delta_acx * dist_ac - delta_acx * dist_ab) * 0.5f / N; 

                if(dist_ab != 0 && dist_ac != 0 && N != 0)
                {
                    return x * x + y * y;
                }
                else
                {
                    return -1.0f;
                } 
            }
        };
        
        __host__ __device__ 
        void circumCenter(    const float& Ax,
                              const float& Ay,
                              const float& Bx,
                              const float& By,
                              const float& Cx,
                              const float& Cy,
                              float& x, 
                              float& y); 
    }       
}

#endif // CIRCUMCIRCLE_CUH_INCLUDED
