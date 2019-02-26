// NonMaximumSupression.hpp 
#ifndef NONMAXIMUMSUPRESSION_H_INCLUDED
#define NONMAXIMUMSUPRESSION_H_INCLUDED


#include <vector>

namespace ced 
{
    namespace cpu
    {
        void nonMaximumSupression(int& _height, int& _width, std::vector<float> _orientation, std::vector<float>& _pixelData);
    }
}
#endif //NONMAXIMUMSUPRESSION_H_INCLUDED


