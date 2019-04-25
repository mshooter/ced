#include "AssignColToTri.hpp"

namespace ced
{
    namespace cpu
    {
        void assignColourToTri( std::vector<float>& red, 
                                std::vector<float>& green,
                                std::vector<float>& blue,
                                const std::vector<unsigned int>& pixIDs,
                                const float& r,
                                const float& g, 
                                const float& b)
        {
            for(auto const& id : pixIDs)
            {
                red[id] = r; 
                green[id] = g;
                blue[id] = b;
            }
        }
    }
}
