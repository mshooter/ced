#include "GetWhitePixels.cuh"

#include "Math.cuh"
#include <thrust/iterator/zip_iterator.h>

namespace ced
{
    namespace gpu
    {
        void getWhitePixels(    thrust::device_vector<int>& _whitePixels,
                                thrust::device_vector<float> _red,
                                thrust::device_vector<float> _green,
                                thrust::device_vector<float> _blue
                          )
        {
            // intensity
            thrust::device_vector<float> d_pixItensity(_red.size());

            thrust::transform(  
                                thrust::make_zip_iterator(  thrust::make_tuple(_red.begin(), _green.begin(), _blue.begin())),
                                thrust::make_zip_iterator(  thrust::make_tuple(_red.end(), _green.end(), _blue.end())),
                                d_pixItensity.begin(),
                                add_three_vectors());
            // dummy vector
            thrust::device_vector<int> isWhite_vector(_red.size(), 0);
            thrust::transform(d_pixItensity.begin(), d_pixItensity.end(), isWhite_vector.begin(), isWhite()); 

            thrust::device_vector<int> d_white_pixels(3,-1);
            thrust::device_vector<int> d_indices(3);
            thrust::sequence(d_indices.begin(), d_indices.end());
            thrust::transform(  thrust::make_zip_iterator( thrust::make_tuple(isWhite_vector.begin(), d_indices.begin())),
                                thrust::make_zip_iterator( thrust::make_tuple(isWhite_vector.end(), d_indices.end())),
                                d_white_pixels.begin(),
                                isIdentity());    
                }

            //std::vector<int> white_pix_host(3);

            //auto itr = thrust::remove_if(white_pixels.begin(), white_pixels.end(), isNegative()); 
            //int size = thrust::distance(white_pixels.begin(), itr);

            //white_pix_host.resize(size);

            // thrust::copy(white_pixels.begin(), itr, white_pix_host.begin());
    }
}


