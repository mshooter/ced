#include "Hysterysis.cuh"
#include "ThrustFunctors.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h> 
#include <thrust/transform.h> 
#include "Math.cuh"

namespace ced
{
    namespace gpu
    {
        thrust::device_vector<int> hysterysis(    thrust::device_vector<int>& _edge_ids,
                            thrust::device_vector<float>& _magnitude,
                            float _min_value,
                            float _max_value,
                            int _width
                        )
        {
            
            // create the neighbours 
            int amountOfpixels = _magnitude.size();
            thrust::device_vector<int> d_ni (amountOfpixels);
            thrust::device_vector<int> d_nei(amountOfpixels);
            thrust::device_vector<int> d_nwi(amountOfpixels);

            thrust::device_vector<int> d_si (amountOfpixels);
            thrust::device_vector<int> d_sei(amountOfpixels);
            thrust::device_vector<int> d_swi(amountOfpixels);

            thrust::device_vector<int> d_ei (amountOfpixels);
            thrust::device_vector<int> d_wi (amountOfpixels);

            thrust::device_vector<int> d_nv (amountOfpixels);
            thrust::device_vector<int> d_nev(amountOfpixels);
            thrust::device_vector<int> d_nwv(amountOfpixels);

            thrust::device_vector<int> d_sv (amountOfpixels);
            thrust::device_vector<int> d_sev(amountOfpixels);
            thrust::device_vector<int> d_swv(amountOfpixels);

            thrust::device_vector<int> d_ev (amountOfpixels);
            thrust::device_vector<int> d_wv (amountOfpixels);

            thrust::transform(  _edge_ids.begin(), 
                                _edge_ids.end(), 
                                thrust::make_zip_iterator(
                                                thrust::make_tuple(d_ni.begin(), d_nei.begin(), d_nwi.begin(), d_si.begin(), d_sei.begin(), d_swi.begin(), d_ei.begin(), d_wi.begin())),
                                find_neighbours(_width));
            // get the values of the neighbours
            typedef thrust::device_vector<float>::iterator valueItr;
            typedef thrust::device_vector<int>::iterator indexItr;
            thrust::permutation_iterator<valueItr, indexItr> c_itr (_magnitude.begin(), _edge_ids.begin());
            thrust::permutation_iterator<valueItr, indexItr> n_itr (_magnitude.begin(), d_ni.begin());
            thrust::permutation_iterator<valueItr, indexItr> ne_itr(_magnitude.begin(), d_nei.begin());
            thrust::permutation_iterator<valueItr, indexItr> nw_itr(_magnitude.begin(), d_nwi.begin());
            thrust::permutation_iterator<valueItr, indexItr> s_itr (_magnitude.begin(), d_si.begin());
            thrust::permutation_iterator<valueItr, indexItr> se_itr(_magnitude.begin(), d_sei.begin());
            thrust::permutation_iterator<valueItr, indexItr> sw_itr(_magnitude.begin(), d_swi.begin());
            thrust::permutation_iterator<valueItr, indexItr> e_itr (_magnitude.begin(), d_ei.begin());
            thrust::permutation_iterator<valueItr, indexItr> w_itr (_magnitude.begin(), d_wi.begin());
            // transform and check 
            int amountOfPoints = _edge_ids.size();
            thrust::device_vector<int> d_isEdge(amountOfPoints, 0);
            thrust::transform(
                        thrust::make_zip_iterator(
                                        thrust::make_tuple( c_itr, n_itr, ne_itr, nw_itr, s_itr, se_itr, sw_itr, e_itr, w_itr)),
                        thrust::make_zip_iterator(
                                        thrust::make_tuple( c_itr + amountOfPoints,
                                                            n_itr + amountOfPoints,
                                                            ne_itr + amountOfPoints,
                                                            nw_itr + amountOfPoints, 
                                                            s_itr + amountOfPoints, 
                                                            se_itr + amountOfPoints, 
                                                            sw_itr + amountOfPoints, 
                                                            e_itr + amountOfPoints, 
                                                            w_itr + amountOfPoints)),
                        d_isEdge.begin(),
                        is_thinEdge(_min_value, _max_value));
            int amountOfEdgePixels = thrust::count(d_isEdge.begin(), d_isEdge.end(), 1);
            thrust::device_vector<int> d_indicesOfEdges(amountOfEdgePixels);
            thrust::copy_if(_edge_ids.begin(), _edge_ids.end(), d_isEdge.begin(), d_indicesOfEdges.begin(), is_identity());
            return d_indicesOfEdges;
        } 
    }
}
