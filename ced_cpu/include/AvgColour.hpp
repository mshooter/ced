#ifndef AVGCOLOUR_H_INCLUDED
#define AVGCOLOUR_H_INCLUDED

#include <vector>

namespace ced
{
    namespace cpu
    {
        void avgColour( const std::vector<float>& red, 
                        const std::vector<float>& green,
                        const std::vector<float>& blue,
                        const std::vector<int>& pixIds, 
                        float& r,
                        float& g, 
                        float& b);
    }
}


#endif // AVGCOLOUR_H_INCLUDED
