#ifndef AVGCOLOUR_H_INCLUDED
#define AVGCOLOUR_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        void avgColour( const std::vector<float>& _imgData, 
                        const std::vector<unsigned int>& pixIds, 
                        float& r,
                        float& g, 
                        float& b);
    }
}


#endif // AVGCOLOUR_H_INCLUDED
