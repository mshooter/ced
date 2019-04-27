// TODO check how to continue

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


struct isStart_tri 
{
    __host__ __device__
    int operator()(const int& i)
    {
        if(i%3 == 0)
        {
            return 1;
        } 
        else
        {
            return 0;
        }
    }
};

struct createIndex
{
    const int width;
    createIndex(int _width) : width(_width) {}
    __host__ __device__
    int operator()(const thrust::tuple<float, float>& t)
    {
        return width * thrust::get<1>(t) + thrust::get<0>(t);
    }
};

TEST(AssignPixToTri, createDummyVector)
{
    int amountOfTriIDs = 12;
    int AmountOfTriangles = amountOfTriIDs / 3;
    // host 
    int amountOfPixels = 6;
    // coordinates
    std::vector<float> h_x = {0.0f, 2.0f, 2.0f, 4.0f, 4.0f, 0.0f};
    std::vector<float> h_y = {2.0f, 2.0f, 0.0f, 2.0f, 0.0f, 0.0f};
    std::vector<int> h_indices = {0,1,2,0,2,5,1,3,2,3,4,2};
    // device
    thrust::device_vector<int> dummy_vector(amountOfTriIDs);

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;
    thrust::device_vector<float> d_indices = h_indices;
    // transform 
    thrust::transform(thrust::make_counting_iterator(3), thrust::make_counting_iterator(amountOfTriIDs), dummy_vector.begin()+3, isStart_tri());
    EXPECT_EQ(dummy_vector[0], 0);
    EXPECT_EQ(dummy_vector[1], 0);
    EXPECT_EQ(dummy_vector[2], 0);
    EXPECT_EQ(dummy_vector[3], 1);
    EXPECT_EQ(dummy_vector[4], 0);
    EXPECT_EQ(dummy_vector[5], 0);
    EXPECT_EQ(dummy_vector[6], 1);
    EXPECT_EQ(dummy_vector[7], 0);
    EXPECT_EQ(dummy_vector[8], 0);
    EXPECT_EQ(dummy_vector[9], 1);
    EXPECT_EQ(dummy_vector[10], 0);
    EXPECT_EQ(dummy_vector[11], 0);
    // inclusive scan to assign each pixel to a triangle 
    thrust::device_vector<int> d_keys(amountOfTriIDs);
    thrust::inclusive_scan(dummy_vector.begin(), dummy_vector.end(), d_keys.begin());
    EXPECT_EQ(d_keys[0], 0);
    EXPECT_EQ(d_keys[1], 0);
    EXPECT_EQ(d_keys[2], 0);

    EXPECT_EQ(d_keys[3], 1);
    EXPECT_EQ(d_keys[4], 1);
    EXPECT_EQ(d_keys[5], 1);

    EXPECT_EQ(d_keys[6], 2);
    EXPECT_EQ(d_keys[7], 2);
    EXPECT_EQ(d_keys[8], 2);

    EXPECT_EQ(d_keys[9], 3);
    EXPECT_EQ(d_keys[10], 3);
    EXPECT_EQ(d_keys[11], 3);
    // get the coordinates of all the triangles  
    // create a zip iterator
    thrust::device_vector<float> d_xCoordinates(amountOfTriIDs); 
    thrust::device_vector<float> d_yCoordinates(amountOfTriIDs); 
    thrust::copy(   thrust::make_zip_iterator(  thrust::make_tuple(
                                                            thrust::make_permutation_iterator(d_x.begin(), d_indices.begin()),
                                                            thrust::make_permutation_iterator(d_y.begin(), d_indices.begin()))
                                             ),
                    thrust::make_zip_iterator(  thrust::make_tuple(
                                                            thrust::make_permutation_iterator(d_x.end(), d_indices.end()),
                                                            thrust::make_permutation_iterator(d_y.end(), d_indices.end()))
                                             ),
                    thrust::make_zip_iterator(  thrust::make_tuple(
                                                            d_xCoordinates.begin(), 
                                                            d_yCoordinates.begin())
                                             )
                );

    EXPECT_FLOAT_EQ(d_xCoordinates[0], 0.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[1], 2.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[2], 2.0f); 

    EXPECT_FLOAT_EQ(d_xCoordinates[3], 0.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[4], 2.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[5], 0.0f); 

    EXPECT_FLOAT_EQ(d_xCoordinates[6], 2.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[7], 4.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[8], 2.0f); 

    EXPECT_FLOAT_EQ(d_xCoordinates[9], 4.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[10], 4.0f); 
    EXPECT_FLOAT_EQ(d_xCoordinates[11], 2.0f); 

    EXPECT_FLOAT_EQ(d_yCoordinates[0], 2.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[1], 2.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[2], 0.0f); 

    EXPECT_FLOAT_EQ(d_yCoordinates[3], 2.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[4], 0.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[5], 0.0f); 

    EXPECT_FLOAT_EQ(d_yCoordinates[6], 2.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[7], 2.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[8], 0.0f); 

    EXPECT_FLOAT_EQ(d_yCoordinates[9], 2.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[10], 0.0f); 
    EXPECT_FLOAT_EQ(d_yCoordinates[11], 0.0f); 
    // find the minimum x, y and maximum x, y, for each triangle 
    // reduce by key 
    thrust::device_vector<int> x_minkeys(AmountOfTriangles);
    thrust::device_vector<float> x_min(AmountOfTriangles); 

    thrust::device_vector<int> y_minkeys(AmountOfTriangles);
    thrust::device_vector<float> y_min(AmountOfTriangles); 

    thrust::device_vector<int> x_maxkeys(AmountOfTriangles);
    thrust::device_vector<float> x_max(AmountOfTriangles); 

    thrust::device_vector<int> y_maxkeys(AmountOfTriangles);
    thrust::device_vector<float> y_max(AmountOfTriangles); 

    // keys and values, keys, and values
    thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_yCoordinates.begin(), y_minkeys.begin(), y_min.begin(), thrust::equal_to<float>(), thrust::minimum<float>());    
    thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_yCoordinates.begin(), y_maxkeys.begin(), y_max.begin(), thrust::equal_to<float>(), thrust::maximum<float>());    
    thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_xCoordinates.begin(), x_minkeys.begin(), x_min.begin(), thrust::equal_to<float>(), thrust::minimum<float>());    
    thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_xCoordinates.begin(), x_maxkeys.begin(), x_max.begin(), thrust::equal_to<float>(), thrust::maximum<float>());    

    EXPECT_FLOAT_EQ(y_min[0], 0.0f);
    EXPECT_FLOAT_EQ(y_min[1], 0.0f);
    EXPECT_FLOAT_EQ(y_min[2], 0.0f);
    EXPECT_FLOAT_EQ(y_min[3], 0.0f);
    
    EXPECT_FLOAT_EQ(y_max[0], 2.0f);
    EXPECT_FLOAT_EQ(y_max[1], 2.0f);
    EXPECT_FLOAT_EQ(y_max[2], 2.0f);
    EXPECT_FLOAT_EQ(y_max[3], 2.0f);

    EXPECT_FLOAT_EQ(x_min[0], 0.0f);
    EXPECT_FLOAT_EQ(x_min[1], 0.0f);
    EXPECT_FLOAT_EQ(x_min[2], 2.0f);
    EXPECT_FLOAT_EQ(x_min[3], 2.0f);
    
    EXPECT_FLOAT_EQ(x_max[0], 2.0f);
    EXPECT_FLOAT_EQ(x_max[1], 2.0f);
    EXPECT_FLOAT_EQ(x_max[2], 4.0f);
    EXPECT_FLOAT_EQ(x_max[3], 4.0f);
    
    // with the min and max you can iterate over the pixels you want, do in kernel ??  
    thrust::device_vector<int> start_index(amountOfTris); 
    thrust::device_vector<int> end_index(amountOfTris); 
    //thrust::transform( thrust::make_zip_iterator( thrust::make_tuple(   x_min.begin(), y_min.begin(), x_min.end(), start_index.end(), ); 
}
