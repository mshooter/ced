#include "AvgColour.hpp"

namespace ced
{
    namespace cpu
    {
        void avgColour( const std::vector<float>& _imgData, 
                        const std::vector<unsigned int>& pixIds, 
                        float& r,
                        float& g, 
                        float& b)
        {
            float pixIDSize = pixIds.size();
            for(auto& id : pixIds)
            {
                r += _imgData[id * 3 + 0];
                g += _imgData[id * 3 + 1];  
                b += _imgData[id * 3 + 2];

            }
            r /= pixIDSize;
            g /= pixIDSize;
            b /= pixIDSize;
        }
    }
}
