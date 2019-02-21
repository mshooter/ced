// NonMaximumSupression.hpp 
#ifndef __NONMAXIMUMSUPRESSION_H_INCLUDED__
#define __NONMAXIMUMSUPRESSION_H_INCLUDED__


#include <vector>

namespace ced 
{
    void nonMaximumSupression(int& _height, int& _width, std::vector<float> _orientation, std::vector<float>& _pixelData);
}
#endif //__NONMAXIMUMSUPRESSION_H_INCLUDED__


