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
#include "AssignColToPix.cuh"
TEST(AssignColToPix, functionLib)
{
    std::vector<int> h_triID = {0,0,1,2,2,3,3};    
    std::vector<int> h_pixID = {5,6,0,1,2,10,9};
    std::vector<float> h_red    = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    std::vector<float> h_green  = {0.2f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    std::vector<float> h_blue   = {0.3f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    int amountOfTris = 4;

    thrust::device_vector<int> d_triID = h_triID;
    thrust::device_vector<int> d_pixID = h_pixID;
    thrust::device_vector<float> d_red      = h_red;
    thrust::device_vector<float> d_green    = h_green;
    thrust::device_vector<float> d_blue     = h_blue;

    ced::gpu::assignColToPix(   d_red,
                                d_green,
                                d_blue,
                                d_triID,
                                d_pixID,
                                amountOfTris);
    
    thrust::copy(d_red.begin(), d_red.end(), h_red.begin());
    thrust::copy(d_green.begin(), d_green.end(), h_green.begin());
    thrust::copy(d_blue.begin(), d_blue.end(), h_blue.begin());
    EXPECT_FLOAT_EQ(h_red[0], 0.1f);
    EXPECT_FLOAT_EQ(h_red[1], 0.25f);
    EXPECT_FLOAT_EQ(h_red[2], 0.25f);
    EXPECT_FLOAT_EQ(h_red[3], 0.4f);
    EXPECT_FLOAT_EQ(h_red[4], 0.5f);
    EXPECT_FLOAT_EQ(h_red[5], 0.7f);
    EXPECT_FLOAT_EQ(h_red[6], 0.7f);
    EXPECT_FLOAT_EQ(h_red[7], 0.9f);
    EXPECT_FLOAT_EQ(h_red[8], 1.0f);
    EXPECT_FLOAT_EQ(h_red[9], 1.15f);
    EXPECT_FLOAT_EQ(h_red[10], 1.15f);

    EXPECT_FLOAT_EQ(h_green[0], 0.2f);
    EXPECT_FLOAT_EQ(h_green[1], 0.25f);
    EXPECT_FLOAT_EQ(h_green[2], 0.25f);
    EXPECT_FLOAT_EQ(h_green[3], 0.4f);
    EXPECT_FLOAT_EQ(h_green[4], 0.5f);
    EXPECT_FLOAT_EQ(h_green[5], 0.7f);
    EXPECT_FLOAT_EQ(h_green[6], 0.7f);
    EXPECT_FLOAT_EQ(h_green[7], 0.9f);
    EXPECT_FLOAT_EQ(h_green[8], 1.0f);
    EXPECT_FLOAT_EQ(h_green[9], 1.15f);
    EXPECT_FLOAT_EQ(h_green[10], 1.15f);

    EXPECT_FLOAT_EQ(h_blue[0], 0.3f);
    EXPECT_FLOAT_EQ(h_blue[1], 0.25f);
    EXPECT_FLOAT_EQ(h_blue[2], 0.25f);
    EXPECT_FLOAT_EQ(h_blue[3], 0.4f);
    EXPECT_FLOAT_EQ(h_blue[4], 0.5f);
    EXPECT_FLOAT_EQ(h_blue[5], 0.7f);
    EXPECT_FLOAT_EQ(h_blue[6], 0.7f);
    EXPECT_FLOAT_EQ(h_blue[7], 0.9f);
    EXPECT_FLOAT_EQ(h_blue[8], 1.0f);
    EXPECT_FLOAT_EQ(h_blue[9], 1.15f);
    EXPECT_FLOAT_EQ(h_blue[10], 1.15f);
}
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
struct divideZip
{
    __host__ __device__
    thrust::tuple<float, float, float> operator()(const thrust::tuple<float, float, float>& t, const float& d)
    {
        thrust::tuple <float, float, float> result;
        thrust::get<0>(result) = thrust::get<0>(t)/ d;
        thrust::get<1>(result) = thrust::get<1>(t)/ d;
        thrust::get<2>(result) = thrust::get<2>(t)/ d;
        return result;
    }
};
#include <thrust/reduce.h>
TEST(AssignColToPix, makeFunctionBetterWithThrust)
{
    std::vector<int> h_triID = {0,0,1,2,2,3,3};    
    std::vector<int> h_pixID = {5,6,0,1,2,10,9};
    std::vector<float> h_red    = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    std::vector<float> h_green  = {0.2f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    std::vector<float> h_blue   = {0.3f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    thrust::device_vector<int> d_triID = h_triID;
    thrust::device_vector<int> d_pixID = h_pixID;
    thrust::device_vector<float> d_red      = h_red;
    thrust::device_vector<float> d_green    = h_green;
    thrust::device_vector<float> d_blue     = h_blue;

    int amountOfPixelsReplaced = h_pixID.size();

    // permutation iterator 
    thrust::device_vector<float> d_result_red(amountOfPixelsReplaced);
    thrust::device_vector<float> d_result_green(amountOfPixelsReplaced);
    thrust::device_vector<float> d_result_blue(amountOfPixelsReplaced);

    typedef thrust::device_vector<float>::iterator valueItr;
    typedef thrust::device_vector<int>::iterator indexItr;

    thrust::permutation_iterator<valueItr, indexItr> red_itr(d_red.begin(), d_pixID.begin());
    thrust::permutation_iterator<valueItr, indexItr> green_itr(d_green.begin(), d_pixID.begin());
    thrust::permutation_iterator<valueItr, indexItr> blue_itr(d_blue.begin(), d_pixID.begin());

    // copy
    thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(  red_itr, 
                                                                green_itr, 
                                                                blue_itr)),
                 thrust::make_zip_iterator(thrust::make_tuple(  red_itr+amountOfPixelsReplaced, 
                                                                green_itr+amountOfPixelsReplaced, 
                                                                blue_itr+amountOfPixelsReplaced)),
                 thrust::make_zip_iterator(thrust::make_tuple(  d_result_red.begin(),
                                                                d_result_green.begin(),
                                                                d_result_blue.begin())));

    std::vector<float> h_result_red(d_result_red.begin(), d_result_red.end()); 
    std::vector<float> h_result_green(d_result_green.begin(), d_result_green.end()); 
    std::vector<float> h_result_blue(d_result_blue.begin(), d_result_blue.end()); 
    ASSERT_EQ(h_result_red[0], 0.6f);
    ASSERT_EQ(h_result_red[1], 0.8f);
    ASSERT_EQ(h_result_red[2], 0.1f);
    ASSERT_EQ(h_result_red[3], 0.2f);
    ASSERT_EQ(h_result_red[4], 0.3f);
    ASSERT_EQ(h_result_red[5], 1.2f);
    ASSERT_EQ(h_result_red[6], 1.1f);
    
    ASSERT_EQ(h_result_green[0], 0.6f);
    ASSERT_EQ(h_result_green[1], 0.8f);
    ASSERT_EQ(h_result_green[2], 0.2f);
    ASSERT_EQ(h_result_green[3], 0.2f);
    ASSERT_EQ(h_result_green[4], 0.3f);
    ASSERT_EQ(h_result_green[5], 1.2f);
    ASSERT_EQ(h_result_green[6], 1.1f);

    ASSERT_EQ(h_result_blue[0], 0.6f);
    ASSERT_EQ(h_result_blue[1], 0.8f);
    ASSERT_EQ(h_result_blue[2], 0.3f);
    ASSERT_EQ(h_result_blue[3], 0.2f);
    ASSERT_EQ(h_result_blue[4], 0.3f);
    ASSERT_EQ(h_result_blue[5], 1.2f);
    ASSERT_EQ(h_result_blue[6], 1.1f);

    // for all the values of pixels reduce by key
    int amountOfTri = 4;
    thrust::device_vector<float> d_red_sum(amountOfTri);
    thrust::device_vector<float> d_red_ksum(amountOfTri);
    thrust::device_vector<float> d_green_sum(amountOfTri);
    thrust::device_vector<float> d_green_ksum(amountOfTri);
    thrust::device_vector<float> d_blue_sum(amountOfTri);
    thrust::device_vector<float> d_blue_ksum(amountOfTri);
    // must do a zip ipterator
    thrust::reduce_by_key(  d_triID.begin(),
                            d_triID.end(),
                            d_result_red.begin(),
                            d_red_ksum.begin(),
                            d_red_sum.begin()
                         );
    thrust::reduce_by_key(  d_triID.begin(),
                            d_triID.end(),
                            d_result_green.begin(),
                            d_green_ksum.begin(),
                            d_green_sum.begin()
                         );
    thrust::reduce_by_key(  d_triID.begin(),
                            d_triID.end(),
                            d_result_blue.begin(),
                            d_blue_ksum.begin(),
                            d_blue_sum.begin()
                         );
    std::vector<float> h_red_sum(d_red_sum.begin(), d_red_sum.end()); 
    std::vector<float> h_green_sum(d_green_sum.begin(), d_green_sum.end()); 
    std::vector<float> h_blue_sum(d_blue_sum.begin(), d_blue_sum.end()); 
    ASSERT_FLOAT_EQ(h_red_sum[0], 1.4f);
    ASSERT_FLOAT_EQ(h_red_sum[1], 0.1f);
    ASSERT_FLOAT_EQ(h_red_sum[2], 0.5f);
    ASSERT_FLOAT_EQ(h_red_sum[3], 2.3f);

    ASSERT_FLOAT_EQ(h_green_sum[0], 1.4f);
    ASSERT_FLOAT_EQ(h_green_sum[1], 0.2f);
    ASSERT_FLOAT_EQ(h_green_sum[2], 0.5f);
    ASSERT_FLOAT_EQ(h_green_sum[3], 2.3f);

    ASSERT_FLOAT_EQ(h_blue_sum[0], 1.4f);
    ASSERT_FLOAT_EQ(h_blue_sum[1], 0.3f);
    ASSERT_FLOAT_EQ(h_blue_sum[2], 0.5f);
    ASSERT_FLOAT_EQ(h_blue_sum[3], 2.3f);

    // how many pixels in every triangle
    // the size should be the image size
    thrust::device_vector<float> d_amountOfPix(4);
    thrust::device_vector<float> d_kamountOfPix(4);
    thrust::reduce_by_key(d_triID.begin(), d_triID.end(), thrust::make_constant_iterator(1), d_kamountOfPix.begin(), d_amountOfPix.begin());
    std::vector<float> h_amountOfPix(d_amountOfPix.begin(), d_amountOfPix.end());
    ASSERT_EQ(h_amountOfPix[0], 2); 
    ASSERT_EQ(h_amountOfPix[1], 1); 
    ASSERT_EQ(h_amountOfPix[2], 2); 
    ASSERT_EQ(h_amountOfPix[3], 2); 

    // divide the amount of pixels by the sum  to get the average
    thrust::transform(  thrust::make_zip_iterator(  thrust::make_tuple( d_red_sum.begin(),
                                                                        d_green_sum.begin(),
                                                                        d_blue_sum.begin())),
                        thrust::make_zip_iterator(  thrust::make_tuple( d_red_sum.end(),
                                                                        d_green_sum.end(),
                                                                        d_blue_sum.end())),
                        d_amountOfPix.begin(), 
                        thrust::make_zip_iterator(  thrust::make_tuple( d_red_sum.begin(),
                                                                        d_green_sum.begin(),
                                                                        d_blue_sum.begin())),
                        divideZip()); 

    thrust::copy(d_red_sum.begin(), d_red_sum.end(), h_red_sum.begin());
    thrust::copy(d_green_sum.begin(), d_green_sum.end(), h_green_sum.begin());
    thrust::copy(d_blue_sum.begin(), d_blue_sum.end(), h_blue_sum.begin());
    ASSERT_FLOAT_EQ(h_red_sum[0], 0.7f); 
    ASSERT_FLOAT_EQ(h_red_sum[1], 0.1f); 
    ASSERT_FLOAT_EQ(h_red_sum[2], 0.25f); 
    ASSERT_FLOAT_EQ(h_red_sum[3], 1.15f); 
    
    ASSERT_FLOAT_EQ(h_green_sum[0], 0.7f); 
    ASSERT_FLOAT_EQ(h_green_sum[1], 0.2f); 
    ASSERT_FLOAT_EQ(h_green_sum[2], 0.25f); 
    ASSERT_FLOAT_EQ(h_green_sum[3], 1.15f); 

    ASSERT_FLOAT_EQ(h_blue_sum[0], 0.7f); 
    ASSERT_FLOAT_EQ(h_blue_sum[1], 0.3f); 
    ASSERT_FLOAT_EQ(h_blue_sum[2], 0.25f); 
    ASSERT_FLOAT_EQ(h_blue_sum[3], 1.15f); 

    // permutation this works - copy the result back to the pixels idx 
    thrust::device_vector<float> d_red_final_pix(7);
    thrust::device_vector<float> d_green_final_pix(7);
    thrust::device_vector<float> d_blue_final_pix(7);

    thrust::copy(
                    thrust::make_zip_iterator(  thrust::make_tuple( thrust::make_permutation_iterator(d_red_sum.begin(), d_triID.begin()),
                                                                    thrust::make_permutation_iterator(d_green_sum.begin(), d_triID.begin()),
                                                                    thrust::make_permutation_iterator(d_blue_sum.begin(), d_triID.begin()))),
                    thrust::make_zip_iterator(  thrust::make_tuple( thrust::make_permutation_iterator(d_red_sum.end(), d_triID.end()),
                                                                    thrust::make_permutation_iterator(d_green_sum.end(), d_triID.end()),
                                                                    thrust::make_permutation_iterator(d_blue_sum.end(), d_triID.end()))), 
                    thrust::make_zip_iterator( thrust::make_tuple(  d_red_final_pix.begin(),
                                                                    d_green_final_pix.begin(),
                                                                    d_blue_final_pix.begin())));

    std::vector<float> h_red_final_pix(d_red_final_pix.begin(), d_red_final_pix.end());
    std::vector<float> h_green_final_pix(d_green_final_pix.begin(), d_green_final_pix.end());
    std::vector<float> h_blue_final_pix(d_blue_final_pix.begin(), d_blue_final_pix.end());

    ASSERT_FLOAT_EQ(h_red_final_pix[0], 0.7f); 
    ASSERT_FLOAT_EQ(h_red_final_pix[1], 0.7f); 
    ASSERT_FLOAT_EQ(h_red_final_pix[2], 0.1f); 
    ASSERT_FLOAT_EQ(h_red_final_pix[3], 0.25f); 
    ASSERT_FLOAT_EQ(h_red_final_pix[4], 0.25f); 
    ASSERT_FLOAT_EQ(h_red_final_pix[5], 1.15f); 
    ASSERT_FLOAT_EQ(h_red_final_pix[6], 1.15f); 

    ASSERT_FLOAT_EQ(h_green_final_pix[0], 0.7f); 
    ASSERT_FLOAT_EQ(h_green_final_pix[1], 0.7f); 
    ASSERT_FLOAT_EQ(h_green_final_pix[2], 0.2f); 
    ASSERT_FLOAT_EQ(h_green_final_pix[3], 0.25f); 
    ASSERT_FLOAT_EQ(h_green_final_pix[4], 0.25f); 
    ASSERT_FLOAT_EQ(h_green_final_pix[5], 1.15f); 
    ASSERT_FLOAT_EQ(h_green_final_pix[6], 1.15f); 

    ASSERT_FLOAT_EQ(h_blue_final_pix[0], 0.7f); 
    ASSERT_FLOAT_EQ(h_blue_final_pix[1], 0.7f); 
    ASSERT_FLOAT_EQ(h_blue_final_pix[2], 0.3f); 
    ASSERT_FLOAT_EQ(h_blue_final_pix[3], 0.25f); 
    ASSERT_FLOAT_EQ(h_blue_final_pix[4], 0.25f); 
    ASSERT_FLOAT_EQ(h_blue_final_pix[5], 1.15f); 
    ASSERT_FLOAT_EQ(h_blue_final_pix[6], 1.15f); 

    // need to assign the values to the red, green, blue iterator 
    thrust::copy(d_red_final_pix.begin(), d_red_final_pix.end(), red_itr);
    thrust::copy(d_green_final_pix.begin(), d_green_final_pix.end(), green_itr);
    thrust::copy(d_blue_final_pix.begin(), d_blue_final_pix.end(), blue_itr);
    ASSERT_FLOAT_EQ(red_itr[0], 0.7f); // 5
    ASSERT_FLOAT_EQ(red_itr[1], 0.7f); // 6 
    ASSERT_FLOAT_EQ(red_itr[2], 0.1f); // 0
    ASSERT_FLOAT_EQ(red_itr[3], 0.25f); // 1
    ASSERT_FLOAT_EQ(red_itr[4], 0.25f); // 2
    ASSERT_FLOAT_EQ(red_itr[5], 1.15f); // 10
    ASSERT_FLOAT_EQ(red_itr[6], 1.15f); // 9
 
    ASSERT_FLOAT_EQ(green_itr[0], 0.7f); // 5
    ASSERT_FLOAT_EQ(green_itr[1], 0.7f); // 6 
    ASSERT_FLOAT_EQ(green_itr[2], 0.2f); // 0
    ASSERT_FLOAT_EQ(green_itr[3], 0.25f); // 1
    ASSERT_FLOAT_EQ(green_itr[4], 0.25f); // 2
    ASSERT_FLOAT_EQ(green_itr[5], 1.15f); // 10
    ASSERT_FLOAT_EQ(green_itr[6], 1.15f); // 9

    ASSERT_FLOAT_EQ(blue_itr[0], 0.7f); // 5
    ASSERT_FLOAT_EQ(blue_itr[1], 0.7f); // 6 
    ASSERT_FLOAT_EQ(blue_itr[2], 0.3f); // 0
    ASSERT_FLOAT_EQ(blue_itr[3], 0.25f); // 1
    ASSERT_FLOAT_EQ(blue_itr[4], 0.25f); // 2
    ASSERT_FLOAT_EQ(blue_itr[5], 1.15f); // 10
    ASSERT_FLOAT_EQ(blue_itr[6], 1.15f); // 9

    // copy the device vector back to host
    thrust::copy(d_red.begin(), d_red.end(), h_red.begin());
    thrust::copy(d_green.begin(), d_green.end(), h_green.begin());
    thrust::copy(d_blue.begin(), d_blue.end(), h_blue.begin());
    EXPECT_FLOAT_EQ(h_red[0], 0.1f);
    EXPECT_FLOAT_EQ(h_red[1], 0.25f);
    EXPECT_FLOAT_EQ(h_red[2], 0.25f);
    EXPECT_FLOAT_EQ(h_red[3], 0.4f);
    EXPECT_FLOAT_EQ(h_red[4], 0.5f);
    EXPECT_FLOAT_EQ(h_red[5], 0.7f);
    EXPECT_FLOAT_EQ(h_red[6], 0.7f);
    EXPECT_FLOAT_EQ(h_red[7], 0.9f);
    EXPECT_FLOAT_EQ(h_red[8], 1.0f);
    EXPECT_FLOAT_EQ(h_red[9], 1.15f);
    EXPECT_FLOAT_EQ(h_red[10], 1.15f);

    EXPECT_FLOAT_EQ(h_green[0], 0.2f);
    EXPECT_FLOAT_EQ(h_green[1], 0.25f);
    EXPECT_FLOAT_EQ(h_green[2], 0.25f);
    EXPECT_FLOAT_EQ(h_green[3], 0.4f);
    EXPECT_FLOAT_EQ(h_green[4], 0.5f);
    EXPECT_FLOAT_EQ(h_green[5], 0.7f);
    EXPECT_FLOAT_EQ(h_green[6], 0.7f);
    EXPECT_FLOAT_EQ(h_green[7], 0.9f);
    EXPECT_FLOAT_EQ(h_green[8], 1.0f);
    EXPECT_FLOAT_EQ(h_green[9], 1.15f);
    EXPECT_FLOAT_EQ(h_green[10], 1.15f);

    EXPECT_FLOAT_EQ(h_blue[0], 0.3f);
    EXPECT_FLOAT_EQ(h_blue[1], 0.25f);
    EXPECT_FLOAT_EQ(h_blue[2], 0.25f);
    EXPECT_FLOAT_EQ(h_blue[3], 0.4f);
    EXPECT_FLOAT_EQ(h_blue[4], 0.5f);
    EXPECT_FLOAT_EQ(h_blue[5], 0.7f);
    EXPECT_FLOAT_EQ(h_blue[6], 0.7f);
    EXPECT_FLOAT_EQ(h_blue[7], 0.9f);
    EXPECT_FLOAT_EQ(h_blue[8], 1.0f);
    EXPECT_FLOAT_EQ(h_blue[9], 1.15f);
    EXPECT_FLOAT_EQ(h_blue[10], 1.15f);
}
