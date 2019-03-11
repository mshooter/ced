#include "Seed.hpp"
#include <cstdlib>

namespace ced
{
    namespace cpu
    {
        int selectSeed(int _min, int _max)
        {
            return (_min + std::rand() % ((_max + 1) - _min));
        }
    }
}
