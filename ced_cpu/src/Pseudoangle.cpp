#include "Pseudoangle.hpp"

namespace ced
{
    namespace cpu
    {
        float pseudo_angle(Point p)
        {
            const float d = p.x / (std::abs(p.x) + std::abs(p.y));
            return (p.y > 0.0f ? 3.0f - d : 1.0f + d) / 4.0f; 
        }
    }
}
