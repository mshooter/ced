#include "CircumCircle.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ __device__ float2 circumCenter(const float2& A, const float2& B, const float2& C)
        {
            const float delta_abx = B.x-A.x;
            const float delta_aby = B.y-A.y;
            const float delta_acx = C.x-A.x;
            const float delta_acy = C.y-A.y;
            
            const float dist_ab = delta_abx * delta_abx + delta_aby * delta_aby;
            const float dist_ac = delta_acx * delta_acx + delta_acy * delta_acy;
            const float N = delta_abx * delta_acy - delta_aby * delta_acx;

            const float x = (delta_acy * dist_ab - delta_aby * dist_ac) * 0.5f / N; 
            const float y = (delta_acx * dist_ac - delta_acx * dist_ab) * 0.5f / N; 
            
            float2 result = make_float2(x,y);
            return result;
        }
    }
}
