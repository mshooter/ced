#include "CCW.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ __device__ 
        float CCW(float2 p1, float2 p2, float2 p3)
        {
            float y_21 = (p2.y - p1.y); 
            float x_32 = (p3.x - p2.x); 
            float x_21 = (p2.x - p1.x); 
            float y_32 = (p3.y - p2.y); 

            float ccw = (y_21 * x_32 - x_21 * y_32);
            // ccw < 0 
            return ccw ;   
        }
    }
}
