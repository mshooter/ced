#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/permutation_iterator.h>

TEST(Gradients, Gx)
{
    // ----------------------host memory allocation--------------------------------------
    // permutation iterator thrust
    // in order values 
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
    std::vector<int> h_indices(5*5*3);
    // how to create the indices? 
    // --------------------device memory allocation--------------------------------------
    thrust::device_vector<float> val;
    // not in order indices
    // how to create the indices
    thrust::device_vector<int> indices;
    typedef thrust::device_vector<float>::iterator elItr;
    typedef thrust::device_vector<int>::iterator indItr;
    thrust::permutatoin_iterator<elmItr, indItr> iter(val.begin(), indices.itr());
    
}
