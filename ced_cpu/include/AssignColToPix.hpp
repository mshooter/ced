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
                std::vector<float>& imgData,
                std::multimap<unsigned int, unsigned int> pixIDdepTri, 
                const unsigned int& amountOfTriangles);
    }
}

#endif // ASSIGNCOLTOPIX_H_INCLUDED
