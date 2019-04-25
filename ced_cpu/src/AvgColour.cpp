#include "AvgColour.hpp"

namespace ced
{
    namespace cpu
    {
        void avgColour( const std::vector<float>& red, 
                        const std::vector<float>& green,
                        const std::vector<float>& blue,
                        const std::vector<unsigned int>& pixIds, 
                        float& r,
                        float& g, 
                        float& b)
        {
            float pixIDSize = pixIds.size();
            for(auto& id : pixIds)
            {
                r += red[id];
                g += green[id];  
                b += blue[id];
            }
            r /= pixIDSize;
            g /= pixIDSize;
            b /= pixIDSize;
        }
    }
}
