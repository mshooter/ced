#include "CCW.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ __device__ 
        float CCW(
                    float p1x, 
                    float p1y, 
                    float p2x,  
                    float p2y, 
                    float p3x, 
                    float p3y)
        {
            float y_21 = (p2y - p1y); 
            float x_32 = (p3x - p2x); 
            float x_21 = (p2x - p1x); 
            float y_32 = (p3y - p2y); 

            float ccw = (y_21 * x_32 - x_21 * y_32);
            // ccw < 0 
            return ccw ;   
        }
    }
}
