#ifndef ASSIGNCOLTOPIX_H_INCLUDED
#define ASSIGNCOLTOPIX_H_INCLUDED

#include <vector>
#include <map>
#include "Point.hpp"


namespace ced
{
    namespace cpu
    {
        void assignColToPix(
                std::vector<float>& red,
                std::vector<float>& green,
                std::vector<float>& blue,
                std::multimap<int, int>& pixIDdepTri, 
                const int& amountOfTriangles);
    }
}

#endif // ASSIGNCOLTOPIX_H_INCLUDED
