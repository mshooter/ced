#include "IsPixInTri.hpp"

namespace ced
{
    namespace cpu
    {
        bool isPixInTri(Point a, Point b, Point c, Point p)
        {
            // compute vectors
            Point v0 = c - a;      
            Point v1 = b - a;      
            Point v2 = p - a;      

            // compute dot product
            int dot00 = dot<int>(v0, v0);
            int dot01 = dot<int>(v1, v0);
            int dot02 = dot<int>(v0, v2);
            int dot11 = dot<int>(v1, v1);
            int dot12 = dot<int>(v1, v2);
            
            // compute barycentric coordinates
            float invDenom = 1.0f/static_cast<float>(dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
            
            return ((u>=0)&&(v>=0)&&(u+v<=1));
        }
    }
}
