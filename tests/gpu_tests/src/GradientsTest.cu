#include "gtest/gtest.h"

#include "GradientsKernel.cuh"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
inline int iDivUp(const unsigned int &a, const unsigned int &b){return (a%b != 0 ? (a/b+1) : (a/b));}
using namespace ced::gpu;

TEST(Gradients, Gx)
{
    int _height = 3; 
    const int nwidth = 3;
    std::vector<float> _pixelData = {0.1f, 0.2f, 0.03f,
                                     0.4f, 0.05f, 0.06f,      
                                     0.7f, 0.08f, 0.09f,      
                                     0.0f, 0.11f, 0.12f,      
                                     0.3f, 0.14f, 0.15f,      

                                     0.16f, 0.31f, 0.41f,      
                                     0.17f, 0.32f, 0.42f,      
                                     0.18f, 0.33f, 0.44f,      
                                     0.19f, 0.34f, 0.45f,      
                                     0.20f, 0.35f, 0.46f,      

                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      

                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      

                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     0.01f, 0.01f, 0.01f,      
                                     };
    // we have the orientations 
    // we have the pixelData
    const std::vector<float> h_kernelX = {-1, 0, 1,
                                          -2, 0, 2,
                                          -1, 0, 1};
    //  ----------------------allocate device mem--------------------------
    const thrust::device_vector<float> d_kernelX = h_kernelX;
    thrust::device_vector<float> d_pixelData = _pixelData;
    thrust::device_vector<float> d_Gx(_height * nwidth);
    // --------------------typecast raw ptr--------------------------------
    float* d_pixelData_ptr = thrust::raw_pointer_cast(&d_pixelData[0]);
    float* d_Gx_ptr = thrust::raw_pointer_cast(&d_Gx[0]);
    const float* d_kernelX_ptr = thrust::raw_pointer_cast(&d_kernelX[0]);
    // --------------------execution config--------------------------------
    int blockW = 32;
    int blockH = 32;
    const dim3 grid(iDivUp(nwidth, blockW),
                    iDivUp(_height, blockH));
    const dim3 threadBlock(blockW, blockH);
    // --------------------calling kernel-------------------------------------
    d_calculateGradients<<<grid, threadBlock>>>(    d_Gx_ptr, 
                                                    d_pixelData_ptr, 
                                                    d_kernelX_ptr,
                                                    nwidth, 
                                                    _height); 
    
    cudaDeviceSynchronize();
    // --------------------convert to host------------------------------------
    std::vector<float> h_Gx(d_Gx.begin(), d_Gx.end());
    std::cout<<"hello"<<std::endl;
    for(auto x : h_Gx)
    {
        std::cout<<x<<std::endl;
    }
}
