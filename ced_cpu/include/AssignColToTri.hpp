#ifndef ASSIGNCOLTOTRI_H_INCLUDED
#define ASSIGNCOLTOTRI_H_INCLUDED

#include <vector>

namespace ced
{
    namespace cpu
    {
        void assignColourToTri( std::vector<float>& red, 
                                std::vector<float>& green,
                                std::vector<float>& blue,
                                const std::vector<int>& pixIDs,
                                const float& r,
                                const float& g, 
                                const float& b);
    }
}

#endif// ASSIGNCOLTOTRI_H_INCLUDED
