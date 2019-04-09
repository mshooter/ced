#ifndef PARAMS_H_INCLUDED
#define PARAMS_H_INCLUDED

#include <limits>

namespace ced
{
    namespace cpu
    {
        constexpr float EPSILON = std::numeric_limits<float>::epsilon();
        constexpr unsigned int INVALID_IDX = std::numeric_limits<unsigned int>::max();
    }
}


#endif // PARAMS_H_INCLUDED
