// Gradients.hpp 
#ifndef GADIENTS_H_INCLUDED
#define GADIENTS_H_INCLUDED

#include <vector>

namespace ced 
{
    namespace cpu
    {
        void calculateGradients(
                int& _height, 
                int& _width, 
                std::vector<float>& _red, 
                std::vector<float>& _green, 
                std::vector<float>& _blue, 
                std::vector<float>& _orientations);
    }
}
#endif //GAUSSIANFILTER_H_INCLUDED

