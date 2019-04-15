#include "AssignColToTri.hpp"

namespace ced
{
    namespace cpu
    {
        void assignColourToTri( std::vector<float>& _imgData, 
                                const std::vector<Point>& pixIdxTri,
                                const float& r,
                                const float& g, 
                                const float& b,
                                const unsigned int& height,
                                const unsigned int& width)
        {
            for(auto t : pixIdxTri)
            {
                _imgData[(t.x + t.y * width) * 3 + 0] = r; 
                _imgData[(t.x + t.y * width) * 3 + 1] = g;
                _imgData[(t.x + t.y * width) * 3 + 2] = b;

            }
        }
    }
}
