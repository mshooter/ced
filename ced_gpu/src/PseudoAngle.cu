#include "PseudoAngle.cuh"

namespace ced
{
    namespace gpu
    {
        __device__ float pseudo_angle(Point p)
        {
            const float d = p.x / (std::abs(p.x) + std::abs(p.y));
            return (p.y > 0.0f ? 3.0f - d : 1.0f + d) / 4.0f; 
        }
    }
}

