#ifndef ASSIGNPIXTOTRI_H_INCLUDED
#define ASSIGNPIXTOTRI_H_INCLUDED

#include <vector>
#include <map>
#include "Point.hpp"

namespace ced
{
    namespace cpu
    {
        void assignPixToTri(
                std::multimap<unsigned int, unsigned int>& pixIDdepTri, 
                const std::vector<unsigned int>& triangleIDs,
                const std::vector<Point>& coordinates, 
                const unsigned int& width
                );
    }
}

#endif // ASSIGNPIXTOTRI_H_INCLUDED
