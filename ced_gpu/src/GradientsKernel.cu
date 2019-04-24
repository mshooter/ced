#include "GradientsKernel.cuh"

namespace ced
{
    namespace gpu
    {
        __global__ void d_calculateGradients(   float* _d_Gx_ptr,
                                                float* _d_pixelData_ptr,
                                                const float* _d_kernelX_ptr,
                                                int _nwidth,
                                                int _nheight)
        {
            const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; 
            const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
            if(j < _nwidth && i < _nheight)
            {
                float sumx = 0.0f;
                for(int y = 0; y < 9; ++y)
                {
                    int w = y % 3;
                    int h = y / 3;
                    int base = ((w+j) + (h+i) * (_nwidth + 2) * 3);
                    float i_pix0 = _d_pixelData_ptr[base + 0];
                    float i_pix1 = _d_pixelData_ptr[base + 1];
                    float i_pix2 = _d_pixelData_ptr[base + 2];
                    //sumx += (i_pix0 + i_pix1 + i_pix2);
                    //printf("%d, %d \n", indexX, indexY);
                    //printf("%d, %d, %d \n", base, base+1, base+2);
                    sumx += (i_pix0 + i_pix1 + i_pix2) * _d_kernelX_ptr[y];
                    //sumx = (i_pix0 + i_pix1 + i_pix2);
                }
                _d_Gx_ptr[j +i * _nwidth] = sumx;
            }
        }
    }
}
