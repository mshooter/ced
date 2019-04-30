#include "gtest/gtest.h"
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <vector>


// orientation, c, n, ne, nw, s, se, sw, e, w
struct is_edge
{
    __host__ __device__ 
    int operator()(const thrust::tuple<float, float, float, float, float, float, float, float, float, float>& t)
    {
        float angle = thrust::get<0>(t);
        float c = thrust::get<1>(t);
        float n = thrust::get<2>(t);
        float ne = thrust::get<3>(t);
        float nw = thrust::get<4>(t);
        float s = thrust::get<5>(t);
        float se = thrust::get<6>(t);
        float sw = thrust::get<7>(t);
        float e = thrust::get<8>(t);
        float w = thrust::get<9>(t);

        if((angle >= 0  && angle <= 22.5f) || (angle >= 157.5f && angle <=180))
        {
            if(c > w && c > e) return 1; 
        }

        else if(angle >= 22.5f  && angle <= 67.5f)
        {
            if(c > nw && c > se) return 1; 
        }

        else if(angle >= 67.5f  && angle <= 112.5f)
        {
            if(c > n && c > s) return 1; 
        }

        else if(angle >= 112.5f  && angle <= 157.5f)
        {
            if(c > nw && c > se) return 1; }
        else
        {
            return 0;
        }
    }
};

struct is_Upper
{
    const float max_value; is_Upper(float _max_value) : max_value(_max_value){}
    __host__ __device__
    int operator()(const float& f) 
    {
        if(f > max_value)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
};

struct isInBound
{
    const int height;
    const int width;
    isInBound(int _h, int _w) : height(_h), width(_w) {}
    __host__ __device__ 
    int operator()(const int& i)
    {
        int w = i % width;
        int h = i / width;
        if(w >= 1 && h >= 1 && w <( width-1) &&( h < height-1))
        {
            return 1; 
        }
        else
        {
            return 0;
        }
    }
};

struct is_identity
{
    __host__ __device__
    bool operator()(const int& t)
    {
        return t == 1;
    }
};

struct find_neighbours 
{
    const int width;
    find_neighbours(int _w) : width(_w) {}
    __host__ __device__ 
    thrust::tuple<int, int, int, int, int, int, int, int> operator()(const int& i)
    {
        thrust::tuple<int, int, int, int, int, int, int, int> result;
        thrust::get<0>(result) = i - width;
        thrust::get<1>(result) = i +1 - width;
        thrust::get<2>(result) = i - 1 - width;
        thrust::get<3>(result) = i + width;
        thrust::get<4>(result) = i + (width+1);
        thrust::get<5>(result) = i + (width-1);
        thrust::get<6>(result) = i + 1;
        thrust::get<7>(result) = i - 1;
        return result;
    }
};

TEST(NonMaxima, UpperValue)
{
    float max_value = 0.5f; 
    int height = 4;
    int width = 4;
    int num_of_pix = width * height;
    
    std::vector<float> h_magnitudes = { 0.1f, 0.5f, 0.4f, 0.8f, 
                                        0.3f, 0.8f, 0.4f, 1.0f, 
                                        1.0f, 2.0f, 0.6f, 0.7f, 
                                        0.8f, 0.9f, 0.7f, 0.8f};
    
    thrust::device_vector<float> d_magnitudes = h_magnitudes; 
    
    // find index with more than max_value
    thrust::device_vector<int> d_dummyUppers(num_of_pix);
    thrust::transform(d_magnitudes.begin(), d_magnitudes.end(), d_dummyUppers.begin(), is_Upper(max_value));
    EXPECT_EQ(d_dummyUppers[0], 0);
    EXPECT_EQ(d_dummyUppers[1], 0);
    EXPECT_EQ(d_dummyUppers[2], 0);
    EXPECT_EQ(d_dummyUppers[3], 1);

    EXPECT_EQ(d_dummyUppers[4], 0);
    EXPECT_EQ(d_dummyUppers[5], 1);
    EXPECT_EQ(d_dummyUppers[6], 0);
    EXPECT_EQ(d_dummyUppers[7], 1);

    EXPECT_EQ(d_dummyUppers[8], 1);
    EXPECT_EQ(d_dummyUppers[9], 1);
    EXPECT_EQ(d_dummyUppers[10], 1);
    EXPECT_EQ(d_dummyUppers[11], 1);

    EXPECT_EQ(d_dummyUppers[12], 1);
    EXPECT_EQ(d_dummyUppers[13], 1);
    EXPECT_EQ(d_dummyUppers[14], 1);
    EXPECT_EQ(d_dummyUppers[15], 1);
    
    // get the indices first and then check  
    int amountOfMaxima = thrust::count(d_dummyUppers.begin(), d_dummyUppers.end(), 1); 
    EXPECT_EQ(amountOfMaxima, 11);
    thrust::device_vector<int> d_indices(num_of_pix);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::device_vector<int> d_maximaIndices(amountOfMaxima, 0);
    thrust::copy_if(d_indices.begin(), d_indices.end(),d_dummyUppers.begin(), d_maximaIndices.begin(), is_identity());
    EXPECT_EQ(d_maximaIndices[0], 3); 
    EXPECT_EQ(d_maximaIndices[1], 5); 
    EXPECT_EQ(d_maximaIndices[2], 7); 
    EXPECT_EQ(d_maximaIndices[3], 8); 
    EXPECT_EQ(d_maximaIndices[4], 9); 
    EXPECT_EQ(d_maximaIndices[5], 10); 
    EXPECT_EQ(d_maximaIndices[6], 11); 
    EXPECT_EQ(d_maximaIndices[7], 12); 
    EXPECT_EQ(d_maximaIndices[8], 13); 
    EXPECT_EQ(d_maximaIndices[9], 14); 
    EXPECT_EQ(d_maximaIndices[10], 15); 
    // check if it is in bound
    thrust::device_vector<int> d_inBound_indices(amountOfMaxima);
    thrust::transform(d_maximaIndices.begin(), d_maximaIndices.end(), d_inBound_indices.begin(), isInBound(height, width));
    EXPECT_EQ(d_inBound_indices[0], 0); 
    EXPECT_EQ(d_inBound_indices[1], 1); 
    EXPECT_EQ(d_inBound_indices[2], 0); 
    EXPECT_EQ(d_inBound_indices[3], 0); 
    EXPECT_EQ(d_inBound_indices[4], 1); 
    EXPECT_EQ(d_inBound_indices[5], 1); 
    EXPECT_EQ(d_inBound_indices[6], 0); 
    EXPECT_EQ(d_inBound_indices[7], 0); 
    EXPECT_EQ(d_inBound_indices[8], 0); 
    EXPECT_EQ(d_inBound_indices[9], 0); 
    EXPECT_EQ(d_inBound_indices[10], 0); 
    // copy those indices
    int nAmountOfMaxima = thrust::count(d_inBound_indices.begin(), d_inBound_indices.end(), 1);
    EXPECT_EQ(nAmountOfMaxima, 3);
    thrust::device_vector<int> d_final_indices(nAmountOfMaxima);
    thrust::copy_if(d_maximaIndices.begin(), d_maximaIndices.end(), d_inBound_indices.begin(), d_final_indices.begin(), is_identity());
    EXPECT_EQ(d_final_indices[0], 5);
    EXPECT_EQ(d_final_indices[1], 9);
    EXPECT_EQ(d_final_indices[2], 10);
    // get orientations also from those indices
    // create their neighbours magnitude list by finding the indices first
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
                        find_neighbours(width));
    EXPECT_EQ(d_ni[0], 1);   EXPECT_EQ(d_nei[0],2);   EXPECT_EQ(d_nwi[0], 0);  EXPECT_EQ(d_si[0], 9);
    EXPECT_EQ(d_ni[1], 5);   EXPECT_EQ(d_nei[1],6);   EXPECT_EQ(d_nwi[1], 4);  EXPECT_EQ(d_si[1], 13);
    EXPECT_EQ(d_ni[2], 6);   EXPECT_EQ(d_nei[2],7);   EXPECT_EQ(d_nwi[2], 5);  EXPECT_EQ(d_si[2], 14);
    
    EXPECT_EQ(d_sei[0], 10); EXPECT_EQ(d_swi[0], 8);  EXPECT_EQ(d_ei[0], 6);   EXPECT_EQ(d_wi[0], 4);
    EXPECT_EQ(d_sei[1], 14); EXPECT_EQ(d_swi[1], 12); EXPECT_EQ(d_ei[1], 10);  EXPECT_EQ(d_wi[1], 8); 
    EXPECT_EQ(d_sei[2], 15); EXPECT_EQ(d_swi[2], 13); EXPECT_EQ(d_ei[2], 11);  EXPECT_EQ(d_wi[2], 9);

    // get the values 
    typedef thrust::device_vector<float>::iterator valueItr;
    typedef thrust::device_vector<int>::iterator indexItr;
    thrust::permutation_iterator<valueItr, indexItr> c_itr(d_magnitudes.begin(), d_final_indices.begin());
    thrust::permutation_iterator<valueItr, indexItr> n_itr(d_magnitudes.begin(), d_ni.begin());
    thrust::permutation_iterator<valueItr, indexItr> ne_itr(d_magnitudes.begin(), d_nei.begin());
    thrust::permutation_iterator<valueItr, indexItr> nw_itr(d_magnitudes.begin(), d_nwi.begin());
    thrust::permutation_iterator<valueItr, indexItr> s_itr(d_magnitudes.begin(), d_si.begin());
    thrust::permutation_iterator<valueItr, indexItr> se_itr(d_magnitudes.begin(), d_sei.begin());
    thrust::permutation_iterator<valueItr, indexItr> sw_itr(d_magnitudes.begin(), d_swi.begin());
    thrust::permutation_iterator<valueItr, indexItr> e_itr(d_magnitudes.begin(), d_ei.begin());
    thrust::permutation_iterator<valueItr, indexItr> w_itr(d_magnitudes.begin(), d_wi.begin());
    EXPECT_EQ(c_itr[0], 0.8f);
    EXPECT_EQ(c_itr[1], 2.0f);
    EXPECT_EQ(c_itr[2], 0.6f);

    EXPECT_EQ(n_itr[0], 0.5f);
    EXPECT_EQ(n_itr[1], 0.8f);
    EXPECT_EQ(n_itr[2], 0.4f);

    EXPECT_EQ(ne_itr[0], 0.4f);
    EXPECT_EQ(ne_itr[1], 0.4f);
    EXPECT_EQ(ne_itr[2], 1.0f);

    EXPECT_EQ(nw_itr[0], 0.1f);
    EXPECT_EQ(nw_itr[1], 0.3f);
    EXPECT_EQ(nw_itr[2], 0.8f);
    // get the orientations
    std::vector<float> h_dummyOrientation = {22.5f, 70.0f, 70.5f}; 
    thrust::device_vector<float> d_dOrientation = h_dummyOrientation;
    thrust::device_vector<int> d_isEdge(nAmountOfMaxima, 0);
    thrust::transform(
                        thrust::make_zip_iterator(
                                            thrust::make_tuple(d_dOrientation.begin(), c_itr, n_itr, ne_itr, nw_itr, s_itr, se_itr, sw_itr, e_itr, w_itr)),
                        thrust::make_zip_iterator(
                                            thrust::make_tuple(d_dOrientation.end(), c_itr+nAmountOfMaxima, n_itr + nAmountOfMaxima, ne_itr + nAmountOfMaxima, nw_itr + nAmountOfMaxima, s_itr + nAmountOfMaxima, se_itr + nAmountOfMaxima, sw_itr + nAmountOfMaxima, e_itr + nAmountOfMaxima, w_itr + nAmountOfMaxima)),
                        d_isEdge.begin(),                      
                        is_edge());
    EXPECT_EQ(d_isEdge[1], 1); 
 
}
