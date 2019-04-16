#include "AssignColToTri.hpp"

namespace ced
{
    namespace cpu
    {
        void assignColourToTri( std::vector<float>& _imgData, 
                                const std::vector<unsigned int>& pixIDs,
                                const float& r,
                                const float& g, 
                                const float& b)
        {
            for(auto const& id : pixIDs)
            {
                _imgData[id*3 + 0] = r; 
                _imgData[id*3 + 1] = g;
                _imgData[id*3 + 2] = b;
            }
        }
    }
}
