#include "FindMidpointTri.hpp"

namespace ced
{
    namespace cpu
    {
        void findMidpointTri(std::vector<Point> _pts, std::vector<unsigned int> _tris, std::vector<Point>& _midpoints )
        {
            for(unsigned int t =0; t < _tris.size(); t+=3)
            {
                Point p0 = _pts[t];
                Point p1 = _pts[t+1];
                Point p2 = _pts[t+2];

                int x = (p0.x + p1.x + p2.x) / 3;
                int y = (p0.y + p1.y + p2.y) / 3;

                _midpoints.push_back(Point(x,y));
            }
        }
    }
}
