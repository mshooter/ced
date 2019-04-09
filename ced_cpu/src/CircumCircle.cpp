#include "CircumCircle.hpp"

namespace ced
{
    namespace cpu
    {
        Point circumCenter(Point A, Point B, Point C)
        {
            Point delta_ab = B-A;
            Point delta_ac = C-A;
            
            const float dist_ab = delta_ab.x * delta_ab.x + delta_ab.y * delta_ab.y;
            const float dist_ac = delta_ac.x * delta_ac.x + delta_ac.y * delta_ac.y;
            const float N = delta_ab.x * delta_ac.y - delta_ab.y * delta_ac.x;
            
            const float x = A.x + (delta_ac.y * dist_ab - delta_ab.y * dist_ac) * 0.5 / N; 
            const float y = A.y + (delta_ac.x * dist_ac - delta_ac.x * dist_ab) * 0.5 / N; 
            
            return Point(x, y);
        }
    }
}

