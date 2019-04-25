#include "AssignColToPix.hpp"
#include "AssignColToTri.hpp"
#include "AvgColour.hpp"
#include <iterator>
#include <algorithm>

namespace ced
{
    namespace cpu
    {
        void assignColToPix(
                std::vector<float>& red,
                std::vector<float>& green,
                std::vector<float>& blue,
                std::multimap<unsigned int, unsigned int>& pixIDdepTri, 
                const unsigned int& amountOfTriangles)
        {
            for(unsigned int t = 0 ; t < amountOfTriangles; ++t)
            {
                std::vector<unsigned int> triPix;
                for(auto const& x : pixIDdepTri)
                {
                    if(t == x.first)
                    {
                        triPix.push_back(x.second);   
                    }
                }
                float r = 0;
                float g = 0;
                float b = 0;
                avgColour(red, green, blue, triPix, r, g, b);
                assignColourToTri(red, green, blue, triPix, r, g, b);
            }
        }
    }
}
