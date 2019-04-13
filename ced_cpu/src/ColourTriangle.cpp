#include "ColourTriangle.hpp"

namespace ced
{
    namespace cpu
    {
        void findMinMax(    const Point& p0, 
                            const Point& p1, 
                            const Point& p2, 
                            const int& width,
                            const int& height,
                            int& minx, 
                            int& maxx, 
                            int& miny, 
                            int& maxy)
        {
            // min max point
            minx = std::min({p0.x, p1.x, p2.x});
            maxx = std::max({p0.x, p1.x, p2.x});
            miny = std::min({p0.y, p1.y, p2.y});
            maxy = std::max({p0.y, p1.y, p2.y});
            // check boundries of image
            minx = std::max(minx, 0);
            miny = std::max(miny, 0);
            maxx = std::min(maxx, width-1);
            maxy = std::min(maxy, height-1);
        }

        void rasterise( const Point& p0,
                        const Point& p1, 
                        const Point& p2,
                        const int& minx, 
                        const int& maxx, 
                        const int& miny, 
                        const int& maxy, 
                        std::vector<Point>& _ptsInsideTri)
        {
            Point p;
            for(p.y = miny; p.y <= maxy; ++p.y)
            {
                for(p.x = minx; p.x <= maxx; ++p.x)
                {
                    // barcentric coordinates
                    float w0 = isCCW<float>(p0, p1, p); 
                    float w1 = isCCW<float>(p1, p2, p); 
                    float w2 = isCCW<float>(p2, p0, p); 
                    // if p is on or inside all edges render pixel
                    if(w0 < 0 || w1 < 0 || w2 < 0)
                    {
                        // do something with pixel
                        _ptsInsideTri.push_back(p);
                    }
                }
            }
        }
    }
}

