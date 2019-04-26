#include "AvgColour.cuh"

#include "AssignAvgColKernel.cuh"
#include "Math.cuh"

#include <thrust/device_vector.h>

namespace ced
{
    namespace gpu
    {   
        void avgColour( float* _red,
                        float* _green,
                        float* _blue,
                        int* _pixIds,
                        float& _r, 
                        float& _g,
                        float& _b,
                        int _amountOfValues,
                        int _amountOfPixels)
        {
            int sizeOfIDs = _amountOfValues;
            // set new device vector
            thrust::device_vector<float> d_nred(_amountOfPixels, 0.0f);
            thrust::device_vector<float> d_ngreen(_amountOfPixels, 0.0f); 
            thrust::device_vector<float> d_nblue(_amountOfPixels, 0.0f);
            // assign the colour to avg
            float* d_nred_ptr    = thrust::raw_pointer_cast(d_nred.data());
            float* d_ngreen_ptr  = thrust::raw_pointer_cast(d_ngreen.data());
            float* d_nblue_ptr   = thrust::raw_pointer_cast(d_nblue.data());

            assignAvgCol<<<1, sizeOfIDs*sizeOfIDs>>>(   _red,

                                                        _green,
                                                        _blue,
                                                        d_nred_ptr,
                                                        d_ngreen_ptr,
                                                        d_nblue_ptr,
                                                        _pixIds,
                                                        sizeOfIDs); 

            cudaDeviceSynchronize();
            // reduce to get the sum to then divide
            _r   = thrust::reduce(d_nred.begin(), d_nred.end());
            _g   = thrust::reduce(d_ngreen.begin(), d_ngreen.end());
            _b   = thrust::reduce(d_nblue.begin(), d_nblue.end());
            //// divide by the amount of pixels
            _r /= static_cast<float>(sizeOfIDs);
            _g /= static_cast<float>(sizeOfIDs);
            _b /= static_cast<float>(sizeOfIDs);
        }
    }
}
