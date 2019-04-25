// NonMaximumSupression.hpp 
#ifndef NONMAXIMUMSUPRESSION_H_INCLUDED
#define NONMAXIMUMSUPRESSION_H_INCLUDED


#include <vector>

namespace ced 
{
    namespace cpu
    {
        void nonMaximumSupression(
                            int& _height, 
                            int& _width, std::vector<float> _orientation, 
                            std::vector<float>& _red,
                            std::vector<float>& _green,
                            std::vector<float>& _blue);
    }
}
#endif //NONMAXIMUMSUPRESSION_H_INCLUDED


