#include "IsPixInTri.hpp"

namespace ced
{
    namespace cpu
    {
        bool isPixInTri(Point a, Point b, Point c, Point p)
        {
            float v_abx = (b.x - a.x);
            float v_aby = (b.y - a.y);

            float v_cax = (c.x - a.x);
            float v_cay = (c.y - a.y);

            float v_acx = (a.x - c.x);
            float v_acy = (a.y - c.y);

            float v_cbx = (c.x - b.x);
            float v_cby = (c.y - b.y);

            float det = v_abx * v_cay - v_aby * v_cax;
            
            float pa_y = (p.y - a.y); 
            float pb_y = (p.y - b.y); 
            float pc_y = (p.y - c.y); 

            float pa_x = (p.x - a.x); 
            float pb_x = (p.x - b.x); 
            float pc_x = (p.x - c.x); 

            float u = det * (v_abx * pa_y - v_aby * pa_x);
            float v = det * (v_cbx * pb_y - v_cby * pb_x);
            float t = det * (v_acx * pc_y - v_acy * pc_x);
            return (u > 0 && v > 0 && t > 0);
        }
    }
}
