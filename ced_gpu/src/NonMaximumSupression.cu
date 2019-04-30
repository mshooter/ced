#include "NonMaximumSupression.cuh"
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <iostream>
namespace ced
{
    namespace gpu
    {
        thrust::device_vector<int> d_nonMaximumSupression(    thrust::device_vector<float>& _directions,
                                        thrust::device_vector<float>& _magnitudes, 
                                        float _max_value,
                                        int _height,
                                        int _width, 
                                        int _num_of_pix)
        {
            // find index with more than max_value - create dummy vector 0 1 0 1 
            thrust::device_vector<int> d_dummyUppers(_num_of_pix);
            thrust::transform(_magnitudes.begin(), _magnitudes.end(), d_dummyUppers.begin(), is_Upper(_max_value));
            // get the indices first (sequence) and then check if it is in the boundry 
            int amountOfMaxima = thrust::count(d_dummyUppers.begin(), d_dummyUppers.end(), 1); 
            thrust::device_vector<int> d_indices(_num_of_pix);
            thrust::sequence(d_indices.begin(), d_indices.end());

            thrust::device_vector<int> d_maximaIndices(amountOfMaxima, 0);
            thrust::copy_if(d_indices.begin(), d_indices.end(),d_dummyUppers.begin(), d_maximaIndices.begin(), is_identity());

            // check if it is in bound
            thrust::device_vector<int> d_inBound_indices(amountOfMaxima);
            thrust::transform(d_maximaIndices.begin(), d_maximaIndices.end(), d_inBound_indices.begin(), isInBound(_height, _width));
            // copy those indices -> so we only have the valid maximum indices
            int nAmountOfMaxima = thrust::count(d_inBound_indices.begin(), d_inBound_indices.end(), 1);
            thrust::device_vector<int> d_final_indices(nAmountOfMaxima);
            thrust::copy_if(d_maximaIndices.begin(), d_maximaIndices.end(), d_inBound_indices.begin(), d_final_indices.begin(), is_identity());


            // create the neighbour indices
            thrust::device_vector<int> d_ni(nAmountOfMaxima);
            thrust::device_vector<int> d_nei(nAmountOfMaxima);
            thrust::device_vector<int> d_nwi(nAmountOfMaxima);

            thrust::device_vector<int> d_si(nAmountOfMaxima);
            thrust::device_vector<int> d_sei(nAmountOfMaxima);
            thrust::device_vector<int> d_swi(nAmountOfMaxima);

            thrust::device_vector<int> d_ei(nAmountOfMaxima);
            thrust::device_vector<int> d_wi(nAmountOfMaxima);

            thrust::device_vector<int> d_nv(nAmountOfMaxima);
            thrust::device_vector<int> d_nev(nAmountOfMaxima);
            thrust::device_vector<int> d_nwv(nAmountOfMaxima);

            thrust::device_vector<int> d_sv(nAmountOfMaxima);
            thrust::device_vector<int> d_sev(nAmountOfMaxima);
            thrust::device_vector<int> d_swv(nAmountOfMaxima);

            thrust::device_vector<int> d_ev(nAmountOfMaxima);
            thrust::device_vector<int> d_wv(nAmountOfMaxima);

            thrust::transform(  d_final_indices.begin(), 
                                d_final_indices.end(), 
                                thrust::make_zip_iterator(
                                                thrust::make_tuple(d_ni.begin(), d_nei.begin(), d_nwi.begin(), d_si.begin(), d_sei.begin(), d_swi.begin(), d_ei.begin(), d_wi.begin())),
                                find_neighbours(_width));
                    

            // get the values of the neighbours 
            typedef thrust::device_vector<float>::iterator valueItr;
            typedef thrust::device_vector<int>::iterator indexItr;
            thrust::permutation_iterator<valueItr, indexItr> c_itr(_magnitudes.begin(), d_final_indices.begin());
            thrust::permutation_iterator<valueItr, indexItr> n_itr(_magnitudes.begin(), d_ni.begin());
            thrust::permutation_iterator<valueItr, indexItr> ne_itr(_magnitudes.begin(), d_nei.begin());
            thrust::permutation_iterator<valueItr, indexItr> nw_itr(_magnitudes.begin(), d_nwi.begin());
            thrust::permutation_iterator<valueItr, indexItr> s_itr(_magnitudes.begin(), d_si.begin());
            thrust::permutation_iterator<valueItr, indexItr> se_itr(_magnitudes.begin(), d_sei.begin());
            thrust::permutation_iterator<valueItr, indexItr> sw_itr(_magnitudes.begin(), d_swi.begin());
            thrust::permutation_iterator<valueItr, indexItr> e_itr(_magnitudes.begin(), d_ei.begin());
            thrust::permutation_iterator<valueItr, indexItr> w_itr(_magnitudes.begin(), d_wi.begin());
            // get the orientations values based on the indices
            thrust::permutation_iterator<valueItr, indexItr> orientation_itr(_directions.begin(), d_final_indices.begin());
            thrust::device_vector<int> d_isEdge(nAmountOfMaxima, 0);
            thrust::transform(
                                thrust::make_zip_iterator(
                                                    thrust::make_tuple(orientation_itr, c_itr, n_itr, ne_itr, nw_itr, s_itr, se_itr, sw_itr, e_itr, w_itr)),
                                thrust::make_zip_iterator(
                                                    thrust::make_tuple( orientation_itr + nAmountOfMaxima, 
                                                                        c_itr+nAmountOfMaxima, 
                                                                        n_itr + nAmountOfMaxima, 
                                                                        ne_itr + nAmountOfMaxima, 
                                                                        nw_itr + nAmountOfMaxima, 
                                                                        s_itr + nAmountOfMaxima, 
                                                                        se_itr + nAmountOfMaxima, 
                                                                        sw_itr + nAmountOfMaxima, 
                                                                        e_itr + nAmountOfMaxima, 
                                                                        w_itr + nAmountOfMaxima)),
                                d_isEdge.begin(),                      
                                is_edge());

            // get all the valid indices
            int amountOfEdgePixels = thrust::count(d_isEdge.begin(), d_isEdge.end(), 1);
            thrust::device_vector<int> d_indicesOfEdges(amountOfEdgePixels);
            thrust::copy_if(d_final_indices.begin(), d_final_indices.end(), d_isEdge.begin(), d_indicesOfEdges.begin(), is_identity());
            return d_indicesOfEdges;
        }
    }
}
