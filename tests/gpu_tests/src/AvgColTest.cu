#include "gtest/gtest.h"

#include "AssignAvgColKernel.cuh"

#include <thrust/device_vector.h>

#include <vector>

TEST(AvgCol, assigningPix)
{
    // ---- host ------
    std::vector<float> h_ored    = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_ogreen  = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_oblue   = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<int>   h_indices= {0,2,4}; 
    int   sizeOfindices = h_indices.size(); 
    // ---- device ------
    thrust::device_vector<float> d_ored = h_ored; 
    thrust::device_vector<float> d_ogreen = h_ogreen; 
    thrust::device_vector<float> d_oblue = h_oblue; 
    
    thrust::device_vector<float> d_nred(h_ored.size(), 0.0f); 
    thrust::device_vector<float> d_ngreen(h_ogreen.size(), 0.0f); 
    thrust::device_vector<float> d_nblue(h_oblue.size(), 0.0f); 

    thrust::device_vector<int> d_indices = h_indices;
    // ---- typecast ------
    float* d_ored_ptr   = thrust::raw_pointer_cast(d_ored.data());
    float* d_ogreen_ptr = thrust::raw_pointer_cast(d_ogreen.data());
    float* d_oblue_ptr  = thrust::raw_pointer_cast(d_oblue.data());

    float* d_nred_ptr   = thrust::raw_pointer_cast(d_nred.data());
    float* d_ngreen_ptr = thrust::raw_pointer_cast(d_ngreen.data());
    float* d_nblue_ptr  = thrust::raw_pointer_cast(d_nblue.data());
    
    int* d_indices_ptr  = thrust::raw_pointer_cast(d_indices.data());

    // ---- call kernel ------
    ced::gpu::assignAvgCol<<<1, sizeOfindices*sizeOfindices>>>(    d_ored_ptr, 
                                                                   d_ogreen_ptr,
                                                                   d_oblue_ptr,
                                                                   d_nred_ptr,
                                                                   d_ngreen_ptr,
                                                                   d_nblue_ptr,
                                                                   d_indices_ptr,
                                                                   sizeOfindices                                                                                        ); 
    // ----- call back to host -------
    thrust::copy(d_nred.begin(), d_nred.end(), h_ored.begin());
    thrust::copy(d_ngreen.begin(), d_ngreen.end(), h_ogreen.begin());
    thrust::copy(d_nblue.begin(), d_nblue.end(), h_oblue.begin());

    // ----- test  -------
    EXPECT_FLOAT_EQ(h_ored[0], 0.01f);
    EXPECT_FLOAT_EQ(h_ored[1], 0.00f);
    EXPECT_FLOAT_EQ(h_ored[2], 0.03f);
    EXPECT_FLOAT_EQ(h_ored[3], 0.00f);
    EXPECT_FLOAT_EQ(h_ored[4], 0.08f);

    EXPECT_FLOAT_EQ(h_ogreen[0], 0.01f);
    EXPECT_FLOAT_EQ(h_ogreen[1], 0.00f);
    EXPECT_FLOAT_EQ(h_ogreen[2], 0.03f);
    EXPECT_FLOAT_EQ(h_ogreen[3], 0.00f);
    EXPECT_FLOAT_EQ(h_ogreen[4], 0.08f);

    EXPECT_FLOAT_EQ(h_oblue[0], 0.01f);
    EXPECT_FLOAT_EQ(h_oblue[1], 0.00f);
    EXPECT_FLOAT_EQ(h_oblue[2], 0.03f);
    EXPECT_FLOAT_EQ(h_oblue[3], 0.00f);
    EXPECT_FLOAT_EQ(h_oblue[4], 0.08f);
}


TEST(AvgCol, reduce)
{
    std::vector<float> h_ored    = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_ogreen  = {0.05f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_oblue   = {0.06f, 0.02f, 0.03f, 0.05f, 0.08f}; 

    thrust::device_vector<float> d_ored     = h_ored;
    thrust::device_vector<float> d_ogreen   = h_ogreen;
    thrust::device_vector<float> d_oblue    = h_oblue;

    float redSum    = thrust::reduce(d_ored.begin()     , d_ored.end());
    float greenSum  = thrust::reduce(d_ogreen.begin()   , d_ogreen.end());
    float blueSum   = thrust::reduce(d_oblue.begin()    , d_oblue.end());

    EXPECT_EQ(redSum, 0.19f);
    EXPECT_EQ(greenSum, 0.23f);
    EXPECT_EQ(blueSum, 0.24f);
}

#include "Math.cuh"
TEST(AvgCol, divideCst)
{
    std::vector<float> h_ored    = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_ogreen  = {0.05f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_oblue   = {0.06f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    
    thrust::device_vector<float> d_ored     = h_ored;
    thrust::device_vector<float> d_ogreen   = h_ogreen;
    thrust::device_vector<float> d_oblue    = h_oblue;

    float redSum    = 0.19f / 3.0f;
    float greenSum  = 0.23f / 3.0f;
    float blueSum   = 0.24f / 3.0f;

    EXPECT_NEAR(redSum  , 0.0633f , 0.01f);
    EXPECT_NEAR(greenSum, 0.076f  , 0.01f);
    EXPECT_NEAR(blueSum , 0.08f   , 0.01f);
}

#include "AvgColour.cuh"
TEST(AvgCol, avgColourFunction)
{
    using namespace ced::gpu; 
    // ---- host ------
    std::vector<float> h_ored    = {0.01f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_ogreen  = {0.05f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<float> h_oblue   = {0.06f, 0.02f, 0.03f, 0.05f, 0.08f}; 
    std::vector<int>   h_indices= {0,2,4}; 
    int amountOfPix = h_ored.size();
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    // --- device -----
    thrust::device_vector<float> d_ored     = h_ored; 
    thrust::device_vector<float> d_ogreen   = h_ogreen; 
    thrust::device_vector<float> d_oblue    = h_oblue; 
    thrust::device_vector<int>   d_indices  = h_indices;
    
    float* d_ored_ptr   = thrust::raw_pointer_cast(d_ored.data());
    float* d_ogreen_ptr = thrust::raw_pointer_cast(d_ogreen.data());
    float* d_oblue_ptr  = thrust::raw_pointer_cast(d_oblue.data());
    int* d_indices_ptr= thrust::raw_pointer_cast(d_indices.data());
    int sizeOfIndices = h_indices.size();
    avgColour(  d_ored_ptr,
                d_ogreen_ptr,
                d_oblue_ptr,
                d_indices_ptr,
                r, 
                g,
                b,
                sizeOfIndices,
                amountOfPix);

    EXPECT_NEAR(r   , 0.04f , 0.01f);
    EXPECT_NEAR(g   , 0.053f  , 0.01f);
    EXPECT_NEAR(b   , 0.056f   , 0.01f);
}
