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
                std::multimap<int, int>& pixIDdepTri, 
                const std::vector<int>& triangleIDs,
                const std::vector<Point>& coordinates, 
                const int& width
                );
    }
}

#endif // ASSIGNPIXTOTRI_H_INCLUDED
