#include "Gradients.cuh"

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>


#include "ImageApplyFilter.cuh"
#include "Math.cuh"

namespace ced
{
    namespace gpu
    {
        void gradients( thrust::device_vector<float>& _gx, 
                        thrust::device_vector<float>& _gy, 
                        thrust::device_vector<float>& _red, 
                        thrust::device_vector<float>& _green, 
                        thrust::device_vector<float>& _blue,
                        int& _height,
                        int& _width)
        {
            std::vector<float> h_kernelX = {-1, 0, 1,
                                            -2, 0, 2,
                                            -1, 0, 1};
     
            std::vector<float> h_kernelY = {-1, -2, -1,
                                             0,  0,  0,
                                             1,  2,  1};
            _height -= 2; _width  -= 2; 
            int nsize = _height*_width;
            // --------------------device allocation----------------------------------
            thrust::device_vector<float> d_kernelX = h_kernelX; 
            thrust::device_vector<float> d_kernelY = h_kernelY; 

            thrust::device_vector<float> _nredx(nsize);
            thrust::device_vector<float> _ngreenx(nsize);
            thrust::device_vector<float> _nbluex(nsize);

            thrust::device_vector<float> _nredy(nsize);
            thrust::device_vector<float> _ngreeny(nsize);
            thrust::device_vector<float> _nbluey(nsize);
            // --------------------typecast raw ptr-----------------------------------
            float* d_kernelXptr = thrust::raw_pointer_cast(d_kernelX.data()); 
            float* d_kernelYptr = thrust::raw_pointer_cast(d_kernelY.data()); 

            float* d_redPtr = thrust::raw_pointer_cast(_red.data());
            float* d_greenPtr = thrust::raw_pointer_cast(_red.data());
            float* d_bluePtr = thrust::raw_pointer_cast(_red.data());

            float* d_nredxPtr = thrust::raw_pointer_cast(_nredx.data());
            float* d_ngreenxPtr = thrust::raw_pointer_cast(_ngreenx.data());
            float* d_nbluexPtr = thrust::raw_pointer_cast(_nbluex.data());

            float* d_nredyPtr = thrust::raw_pointer_cast(_nredy.data());
            float* d_ngreenyPtr = thrust::raw_pointer_cast(_ngreeny.data());
            float* d_nblueyPtr = thrust::raw_pointer_cast(_nbluey.data());
            // --------------------execution config-----------------------------------
            int blockW = 32;
            int blockH = 32;
            const dim3 grid(iDivUp(_width, blockW),
                            iDivUp(_height, blockH));
            const dim3 threadBlock(blockW, blockH);
            // --------------------calling kernel-------------------------------------
            d_applyFilter<<<grid, threadBlock>>>(d_redPtr, 
                                                 d_greenPtr,
                                                 d_bluePtr,
                                                 d_nredxPtr,   
                                                 d_ngreenxPtr,   
                                                 d_nbluexPtr,   
                                                 d_kernelXptr,
                                                 _width,
                                                 _height, 
                                                 3);
            cudaDeviceSynchronize();

            d_applyFilter<<<grid, threadBlock>>>(d_redPtr, 
                                                 d_greenPtr,
                                                 d_bluePtr,
                                                 d_nredyPtr,   
                                                 d_ngreenyPtr,   
                                                 d_nblueyPtr,   
                                                 d_kernelYptr,
                                                 _width,
                                                 _height, 
                                                 3);
            cudaDeviceSynchronize();
            // --------------------copy-------------------------------------
            // get the intensity
            thrust::transform(
                    thrust::make_zip_iterator(
                                        thrust::make_tuple(_nredx.begin(), _ngreenx.begin(), _nbluex.begin())),
                    thrust::make_zip_iterator(
                                        thrust::make_tuple(_nredx.end(), _ngreenx.end(), _nbluex.end())),
                    _gx.begin(), 
                    add_three_vectors());

            thrust::transform(_gx.begin(), _gx.end(), _gx.begin(), divideByConstant<float>(3.0f)); 
            thrust::transform(
                   thrust::make_zip_iterator(
                                       thrust::make_tuple(_nredy.begin(), _ngreeny.begin(), _nbluey.begin())),
                   thrust::make_zip_iterator(
                                       thrust::make_tuple(_nredy.end(), _ngreeny.end(), _nbluey.end())),
                   _gy.begin(), 
                   add_three_vectors());
            thrust::transform(_gy.begin(), _gy.end(), _gy.begin(), divideByConstant<float>(3.0f)); 

        }
    }
}
