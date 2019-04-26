#include "gtest/gtest.h"

#include "AssignColToTriKernel.cuh"

#include <vector>

#include <thrust/device_vector.h>
TEST(AvgCol, assignCol)
{
    // -------host-------
    std::vector<float> h_red    = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_green  = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_blue   = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<int>   h_indices= {0,2,4};
    float c_red     = 1.0f;
    float c_green   = 2.0f;
    float c_blue    = 3.0f;
    int sizeOfIndices = h_indices.size();
    // -------device-------
    thrust::device_vector<float> d_red      = h_red;
    thrust::device_vector<float> d_green    = h_green;
    thrust::device_vector<float> d_blue     = h_blue;
    thrust::device_vector<int>   d_indices  = h_indices;
    // -------typecast-------
    float* d_red_ptr    = thrust::raw_pointer_cast(d_red.data());
    float* d_green_ptr  = thrust::raw_pointer_cast(d_green.data());
    float* d_blue_ptr   = thrust::raw_pointer_cast(d_blue.data());
    int* d_indices_ptr  = thrust::raw_pointer_cast(d_indices.data());
    // -------set up execution-------
    
    // -------call kernel-------
    ced::gpu::d_assignColToTriKernel<<<1,sizeOfIndices*sizeOfIndices>>>(  d_red_ptr,
                                                                          d_green_ptr,
                                                                          d_blue_ptr,
                                                                          d_indices_ptr,
                                                                          c_red,
                                                                          c_green,
                                                                          c_blue,
                                                                          sizeOfIndices);  
    cudaDeviceSynchronize();
    // -------copy back to host-------
    thrust::copy(d_red.begin(), d_red.end(), h_red.begin()); 
    thrust::copy(d_green.begin(), d_green.end(), h_green.begin()); 
    thrust::copy(d_blue.begin(), d_blue.end(), h_blue.begin()); 

    EXPECT_EQ(h_red[0], 1.0f);
    EXPECT_EQ(h_red[1], 0.02f);
    EXPECT_EQ(h_red[2], 1.0f);
    EXPECT_EQ(h_red[3], 0.05f);
    EXPECT_EQ(h_red[4], 1.0f);

    EXPECT_EQ(h_green[0], 2.0f);
    EXPECT_EQ(h_green[1], 0.02f);
    EXPECT_EQ(h_green[2], 2.0f);
    EXPECT_EQ(h_green[3], 0.05f);
    EXPECT_EQ(h_green[4], 2.0f);

    EXPECT_EQ(h_blue[0], 3.0f);
    EXPECT_EQ(h_blue[1], 0.02f);
    EXPECT_EQ(h_blue[2], 3.0f);
    EXPECT_EQ(h_blue[3], 0.05f);
    EXPECT_EQ(h_blue[4], 3.0f);
}

