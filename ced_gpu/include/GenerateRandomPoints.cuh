#ifndef GENERATERANDOMPOINT_CUH_INCLUDED
#define GENERATERANDOMPOINT_CUH_INCLUDED

namespace ced
{
    namespace gpu
    {
        void generateRandomPoints(
                                    thrust::vector<float>& _x,
                                    thrust::vector<float>& _y,
                                    int _amountOfValues,
                                    
                                    ); 
    }
}

#endif // GENERATERANDOMPOINT_CUH_INCLUDED
