#ifndef AVGCOLOUR_CUH_INCLUDED
#define AVGCOLOUR_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        void avgColour( float* _red,
                        float* _green,
                        float* _blue,
                        int*   _pixIds,
                        float& _r, 
                        float& _g,
                        float& _b,
                        int _amountOfValues,
                        int _amountOfPixels);
    }
}


#endif //AVGCOLOUR_CUH_INCLUDED
