#include "SortPoints.hpp"
#include <algorithm>

namespace ced
{
    namespace cpu
    {
        int partition(std::vector<ced::cpu::Point>& pt, int p, int q)
        {
            // element to move pivot point
            ced::cpu::Point piv = pt[p];
            int i = p; 
            for(int j=p+1; j <= q; j++)
            {
                if(pt[j].x < piv.x)
                {
                    i=i+1; 
                    std::swap(pt[i], pt[j]);
                }

                if(pt[j].x == piv.x)
                {
                    if(pt[j].y <= piv.y)
                    {
                        i++;
                        std::swap(pt[i], pt[j]);
                    }
                }
            }
            std::swap(pt[i], pt[p]);
            return i;
        }

        void quickSort(std::vector<ced::cpu::Point>& pt, int lo, int hi)
        {
            int r;
            if(lo < hi)
            {
                r = ced::cpu::partition(pt, lo, hi);
                quickSort(pt, lo, r-1);
                quickSort(pt, r+1, hi);
            }
        }
    }
}
