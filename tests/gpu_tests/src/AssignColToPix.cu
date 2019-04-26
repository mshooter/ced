#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <vector>

// first step (just in case)
TEST(AssignColToPix, sortByKeyTri)
{
    std::vector<int> h_triID = {0,1,0,2,3,3,2};    
    std::vector<int> h_pixID = {5,0,6,1,28,29,2};
    thrust::device_vector<int> d_triID = h_triID; 
    thrust::device_vector<int> d_pixID = h_pixID; 
   
    thrust::sort_by_key(d_triID.begin(), d_triID.end(), d_pixID.begin()); 
    
    //  ----- test --------
    thrust::copy(d_triID.begin(), d_triID.end(), h_triID.begin());
    thrust::copy(d_pixID.begin(), d_pixID.end(), h_pixID.begin());
    
    EXPECT_EQ(h_triID[0],0);
    EXPECT_EQ(h_triID[1],0);
    EXPECT_EQ(h_triID[2],1);
    EXPECT_EQ(h_triID[3],2);
    EXPECT_EQ(h_triID[4],2);
    EXPECT_EQ(h_triID[5],3);
    EXPECT_EQ(h_triID[6],3);

    EXPECT_EQ(h_pixID[0],5);
    EXPECT_EQ(h_pixID[1],6);
    EXPECT_EQ(h_pixID[2],0);
    EXPECT_EQ(h_pixID[3],1);
    EXPECT_EQ(h_pixID[4],2);
    EXPECT_EQ(h_pixID[5],28);
    EXPECT_EQ(h_pixID[6],29);
}
// second step: makr the beginning of each segment 
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
struct isStartNewSeg
{
    __host__ __device__
    int operator()(const thrust::tuple<int, int>& i)
    {
        // if same key return false 
        if(thrust::get<0>(i) != thrust::get<1>(i))
        {   
            return 1;
        }
        else
        {
            return 0;
        }
    }
};
#include <thrust/reduce.h>
TEST(AssignColToPix, makeFunctionBetterWithThrust)
{
    std::vector<int> h_triID = {0,0,1,2,2,3,3};    
    std::vector<int> h_pixID = {5,6,0,1,2,10,9};
    std::vector<float> h_red = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    int sizeOfStartKeys = h_triID.size();
    thrust::device_vector<int> d_triID = h_triID;
    thrust::device_vector<int> d_pixID = h_pixID;
    thrust::device_vector<float> d_red   = h_red;
    // permutation iterator 
    thrust::device_vector<float> d_result(7);
    thrust::copy(thrust::make_permutation_iterator(d_red.begin(), d_pixID.begin()), thrust::make_permutation_iterator(d_red.end(), d_pixID.end()), d_result.begin());
    std::vector<float> h_result(d_result.begin(), d_result.end()); 
    // test
    ASSERT_EQ(h_result[0], 0.6f);
    ASSERT_EQ(h_result[1], 0.8f);
    ASSERT_EQ(h_result[2], 0.1f);
    ASSERT_EQ(h_result[3], 0.2f);
    ASSERT_EQ(h_result[4], 0.3f);
    ASSERT_EQ(h_result[5], 1.2f);
    ASSERT_EQ(h_result[6], 1.1f);
    
    // for all the values of pixels reduce by key
    thrust::device_vector<float> d_sum(4);
    thrust::device_vector<float> d_ksum(4);
    auto new_last = thrust::reduce_by_key(d_triID.begin(), d_triID.end(), d_result.begin(), d_ksum.begin(), d_sum.begin());
    std::vector<float> h_sum(d_sum.begin(), d_sum.end()); 
    ASSERT_FLOAT_EQ(h_sum[0], 1.4f);
    ASSERT_FLOAT_EQ(h_sum[1], 0.1f);
    ASSERT_FLOAT_EQ(h_sum[2], 0.5f);
    ASSERT_FLOAT_EQ(h_sum[3], 2.3f);

    // how many pixels in every triangle
    thrust::device_vector<float> d_amountOfPix(4);
    thrust::device_vector<float> d_kamountOfPix(4);
    auto new_pixels = thrust::reduce_by_key(d_triID.begin(), d_triID.end(), thrust::make_constant_iterator(1), d_kamountOfPix.begin(), d_amountOfPix.begin());
    std::vector<float> h_amountOfPix(d_amountOfPix.begin(), d_amountOfPix.end());
    ASSERT_EQ(h_amountOfPix[0], 2); 
    ASSERT_EQ(h_amountOfPix[1], 1); 
    ASSERT_EQ(h_amountOfPix[2], 2); 
    ASSERT_EQ(h_amountOfPix[3], 2); 

    // divide one by the other to get the average
    thrust::transform(d_sum.begin(), d_sum.end(), d_amountOfPix.begin(), d_sum.begin(), thrust::divides<float>()); 
    thrust::copy(d_sum.begin(), d_sum.end(), h_sum.begin());
    EXPECT_FLOAT_EQ(h_sum[0], 0.7f); 
    EXPECT_FLOAT_EQ(h_sum[1], 0.1f); 
    EXPECT_FLOAT_EQ(h_sum[2], 0.25f); 
    EXPECT_FLOAT_EQ(h_sum[3], 1.15f); 
    
    // permutation this works - tomorrow 
    thrust::device_vector<float> d_final_pix(7);
    thrust::copy(thrust::make_permutation_iterator(d_sum.begin(), d_triID.begin()), thrust::make_permutation_iterator(d_sum.end(), d_triID.end()), d_final_pix.begin());
    std::vector<float> h_final_pix(d_final_pix.begin(), d_final_pix.end());
    
    // need to assign the values to the red
}
