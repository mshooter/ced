#include "AssignColToPix.cuh"
#include "ThrustFunctors.cuh"

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>

namespace ced
{
    namespace gpu
    {
        void assignColToPix(    thrust::device_vector<float>& _d_red,
                                thrust::device_vector<float>& _d_green,
                                thrust::device_vector<float>& _d_blue,
                                thrust::device_vector<int>&  _d_triID,
                                thrust::device_vector<int>&  _d_pixID,
                                int& _amountOfTris)
        {
            // sort by key (triangle ) 
            thrust::sort_by_key(_d_triID.begin(), _d_triID.end(), _d_pixID.begin());
            // the amount of pixels that need to be replaced
            int amountOfPixelsReplaced = _d_pixID.size();
            // create temporary vector (result)
            thrust::device_vector<float> d_result_red(amountOfPixelsReplaced);
            thrust::device_vector<float> d_result_green(amountOfPixelsReplaced);
            thrust::device_vector<float> d_result_blue(amountOfPixelsReplaced);
            // create iterators to later then assign value to the index of the pixels
            typedef thrust::device_vector<float>::iterator valueItr;
            typedef thrust::device_vector<int>::iterator indexItr;
            thrust::permutation_iterator<valueItr, indexItr> red_itr(_d_red.begin(), _d_pixID.begin());
            thrust::permutation_iterator<valueItr, indexItr> green_itr(_d_green.begin(), _d_pixID.begin());
            thrust::permutation_iterator<valueItr, indexItr> blue_itr(_d_blue.begin(), _d_pixID.begin());
            // now we are going to calculate the average therefor we need to copy the iterators into the temporary vectors
             thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(  red_itr, 
                                                                        green_itr, 
                                                                        blue_itr)),
            thrust::make_zip_iterator(thrust::make_tuple(  red_itr+amountOfPixelsReplaced, 
                                                           green_itr+amountOfPixelsReplaced, 
                                                           blue_itr+amountOfPixelsReplaced)),
            thrust::make_zip_iterator(thrust::make_tuple(  d_result_red.begin(),
                                                           d_result_green.begin(),
                                                           d_result_blue.begin())));
            
            // get the sum for each pixl
            // need to assign this ohterwise we cant use function
            thrust::device_vector<float> d_red_sum(_amountOfTris);
            thrust::device_vector<float> d_red_ksum(_amountOfTris);
            thrust::device_vector<float> d_green_sum(_amountOfTris);
            thrust::device_vector<float> d_green_ksum(_amountOfTris);
            thrust::device_vector<float> d_blue_sum(_amountOfTris);
            thrust::device_vector<float> d_blue_ksum(_amountOfTris);
            // right now we are going to do it in the naive way, but maybe we could do a zip iterator, because it is faster 
            thrust::reduce_by_key(  _d_triID.begin(),
                                    _d_triID.end(),
                                    d_result_red.begin(),
                                    d_red_ksum.begin(),
                                    d_red_sum.begin()
                                 );
            thrust::reduce_by_key(  _d_triID.begin(),
                                    _d_triID.end(),
                                    d_result_green.begin(),
                                    d_green_ksum.begin(),
                                    d_green_sum.begin()
                                 );
            thrust::reduce_by_key(  _d_triID.begin(),
                                    _d_triID.end(),
                                    d_result_blue.begin(),
                                    d_blue_ksum.begin(),
                                    d_blue_sum.begin()
                                 );
            // we also need to know how many pixels there is in each triangle 
            // the maximum pixels is the whole image
            thrust::device_vector<float> d_amountOfPix(_d_red.size());
            thrust::device_vector<float> d_kamountOfPix(_d_red.size());
            thrust::reduce_by_key(    _d_triID.begin(), 
                                      _d_triID.end(), 
                                      thrust::make_constant_iterator(1),
                                      d_kamountOfPix.begin(), 
                                      d_amountOfPix.begin());
            // we going to calculate the average for each triangle pixel, by dividing the amount of pixels with the sum
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
            // copy back the average to the pixels ID
            thrust::device_vector<float> d_red_idValuePix(amountOfPixelsReplaced);
            thrust::device_vector<float> d_green_idValuePix(amountOfPixelsReplaced);
            thrust::device_vector<float> d_blue_idValuePix(amountOfPixelsReplaced);
            thrust::copy(
                            thrust::make_zip_iterator(  thrust::make_tuple( thrust::make_permutation_iterator(d_red_sum.begin(), _d_triID.begin()),
                                                                            thrust::make_permutation_iterator(d_green_sum.begin(), _d_triID.begin()),
                                                                            thrust::make_permutation_iterator(d_blue_sum.begin(), _d_triID.begin()))),
                            thrust::make_zip_iterator(  thrust::make_tuple( thrust::make_permutation_iterator(d_red_sum.end(), _d_triID.end()),
                                                                            thrust::make_permutation_iterator(d_green_sum.end(), _d_triID.end()),
                                                                            thrust::make_permutation_iterator(d_blue_sum.end(), _d_triID.end()))), 
                            thrust::make_zip_iterator( thrust::make_tuple(  d_red_idValuePix.begin(),
                                                                    d_green_idValuePix.begin(),
                                                                    d_blue_idValuePix.begin())));
            // need to assign the values we got to the red, green, blue iterators
            thrust::copy(   thrust::make_zip_iterator(  thrust::make_tuple( d_red_idValuePix.begin(),
                                                                            d_green_idValuePix.begin(),
                                                                            d_blue_idValuePix.begin())),
                            thrust::make_zip_iterator(  thrust::make_tuple( d_red_idValuePix.end(),
                                                                            d_green_idValuePix.end(), 
                                                                            d_blue_idValuePix.end())),
                            thrust::make_zip_iterator(  thrust::make_tuple( red_itr,
                                                                            green_itr,
                                                                            blue_itr)));

        }
    }
}
