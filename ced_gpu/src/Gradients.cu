#include "Gradients.cuh"

#include <thrust/device_vector.h>

#include "GradientsKernel.cuh"

namespace ced
{
    namespace gpu
    {
        void calculateGradients(int& _height,
                                int& _width,
                                std::vector<float>& _pixelData,
                                std::vector<float>& _orientations)
        {
            //  -----------------------allocate host mem--------------------------
            _height -= 2; 
            const int nwidth = _width - 2;
            // we have the orientations 
            // we have the pixelData
            const std::vector<float> h_kernelX = {-1, 0, 1,
                                                  -2, 0, 2,
                                                  -1, 0, 1};
     
            const std::vector<float> h_kernelY = {-1, -2, -1,
                                                 0,  0,  0,
                                                 1,  2,  1};
            //  ----------------------allocate device mem--------------------------
            const thrust::device_vector<float> d_kernelX = h_kernelX;
            const thrust::device_vector<float> d_kernelY = h_kernelY;
            thrust::device_vector<float> d_orientations(_height *nwidth);
            thrust::device_vector<float> d_pixelData = _pixelData;
            thrust::device_vector<float> d_Gx(_height * nwidth);
            thrust::device_vector<float> d_Gy(_height * nwidth);
            // --------------------typecast raw ptr--------------------------------
            float* d_orientations_ptr = thrust::raw_pointer_cast(d_orientations.data());
            float* d_pixelData_ptr = thrust::raw_pointer_cast(d_pixelData.data());
            float* d_Gx_ptr = thrust::raw_pointer_cast(&d_Gx[0]);
            float* d_Gy_ptr = thrust::raw_pointer_cast(&d_Gy[0]);
            const float* d_kernelX_ptr = thrust::raw_pointer_cast(d_kernelX.data());
            const float* d_kernelY_ptr = thrust::raw_pointer_cast(d_kernelY.data());
            // --------------------execution config--------------------------------
            int blockW = 32;
            int blockH = 32;
            const dim3 grid(iDivUp(nwidth, blockW),
                            iDivUp(_height, blockH));
            const dim3 threadBlock(blockW, blockH);
            // --------------------calling kernel-------------------------------------
            //d_calculateGradients<<<grid, threadBlock>>>(    d_Gx_ptr, 
            //                                               d_pixelData_ptr, 
            //                                               d_kernelX_ptr,
            //                                               nwidth, 
            //                                               _height); 
            //
            //cudaDeviceSynchronize();
            // --------------------back to host---------------------------------------
            thrust::copy(d_orientations.begin(), d_orientations.end(), _orientations.begin()); 
            // ------------------init back to host------------------------------------
            _pixelData.resize(nwidth * _height * 3);
            _width = std::move(nwidth);
        }
    }
}
