#include "CircumCircle.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ __device__ 
        void circumCenter(
                            const float& Ax, 
                            const float& Ay,
                            const float& Bx,
                            const float& By,
                            const float& Cx,
                            const float& Cy,
                            float& x,
                            float& y)
        {
            const float delta_abx = Bx-Ax;
            const float delta_aby = By-Ay;
            const float delta_acx = Cx-Ax;
            const float delta_acy = Cy-Ay;
            
            const float dist_ab = delta_abx * delta_abx + delta_aby * delta_aby;
            const float dist_ac = delta_acx * delta_acx + delta_acy * delta_acy;
            const float N = delta_abx * delta_acy - delta_aby * delta_acx;

            x = (delta_acy * dist_ab - delta_aby * dist_ac) * 0.5f / N; 
            y = (delta_acx * dist_ac - delta_acx * dist_ab) * 0.5f / N; 
        }
    }
}
