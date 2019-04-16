#ifndef ASSIGNCOLTOTRI_H_INCLUDED
#define ASSIGNCOLTOTRI_H_INCLUDED

#include <vector>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        void assignColourToTri( std::vector<float>& _imgData, 
                                const std::vector<unsigned int>& pixIDs,
                                const float& r,
                                const float& g, 
                                const float& b);
    }
}

#endif// ASSIGNCOLTOTRI_H_INCLUDED
