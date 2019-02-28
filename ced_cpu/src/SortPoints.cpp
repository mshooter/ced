#include "SortPoints.hpp"
#include <algorithm>

namespace ced
{
    namespace cpu
    {
        int partition(std::vector<Point>& _points, int lo, int hi)
        {
            // need to refine the pivotpoint
            Point nP = _points[hi];
            int pivotX = nP.getX(); 
            int i = lo-1;
            for(int j=lo; j <= hi-1; ++j)
            {
                if(_points[j].getX() < pivotX)
                {
                    i++;
                    std::swap(_points[i], _points[j]);
                }            
            }
            std::swap(_points[i+1], nP); 
            return i;
        }
    }
}
