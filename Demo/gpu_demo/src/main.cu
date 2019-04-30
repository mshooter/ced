#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "ParamsImageIO.hpp"

#include "Image.cuh"
#include "GaussianFilter.cuh"
#include "Gradients.cuh"
#include "CalculateMagnitude.cuh"
#include "CalculateDirections.cuh"
#include "NonMaximumSupression.cuh"
#include "Hysterysis.cuh"
#include "ThrustFunctors.cuh"
        struct convertTo2D
        {
            const int width;
            convertTo2D(int _w) : width(_w) {}
            __host__ __device__
            thrust::tuple<int, int> operator()(const thrust::tuple<int,int>&t)
            {
                thrust::tuple<int, int> result;
                thrust::get<0>(result) = int(thrust::get<0>(t)%width);
                thrust::get<1>(result) = thrust::get<1>(t)/width;
                return result; 
                
            }
        }; int main(int argc, char** argv)
{
    const char* filename = argv[1];
   // const char* outgray = argv[2];
   // const char* outgaussian = argv[3];
    //  ------------------------------------init image------------------------------------
    ced::gpu::Image img(filename);
    std::vector<float> originalPixelData = img.getPixelData();
    unsigned int o_width = img.getWidth();
    unsigned int o_height = img.getHeight();
    thrust::device_vector<float> o_red = img.getRedChannel();
    thrust::device_vector<float> o_green = img.getGreenChannel();
    thrust::device_vector<float> o_blue  = img.getBlueChannel();
    // ----------------------------------image -> edge detection--------------------------
    // -----------------------------------gray scale-------------------------------------- 
    img.convertToGrayscale();
    img.saveImage(outgray, true);
    // -----------------------------------gaussian blur-----------------------------------
    std::vector<float> gfilter = ced::gpu::gaussianFilter(5, 1.4f);
    img.applyFilter(gfilter, 5);
    img.saveImage(outgaussian, true);
    int width = img.getWidth();
    int height = img.getHeight();
    int currentAmountOfPixels = width*height;
    thrust::device_vector<float> d_red = img.getRedChannel();
    thrust::device_vector<float> d_green = img.getGreenChannel();
    thrust::device_vector<float> d_blue  = img.getBlueChannel();
    std::vector<float> h_red = img.getRedChannel();
    std::vector<float> h_green = img.getGreenChannel();
    std::vector<float> h_blue  = img.getBlueChannel();
    // -----------------------------------calculate gradients-----------------------------
    // get the gradients
    thrust::device_vector<float> d_gx(currentAmountOfPixels);
    thrust::device_vector<float> d_gy(currentAmountOfPixels);
    ced::gpu::gradients(
                        d_gx, 
                        d_gy, 
                        d_red, 
                        d_green, 
                        d_blue, 
                        height, 
                        width);
    int nsize=height*width;
    thrust::device_vector<float> d_magnitude(nsize);
    ced::gpu::calculateMagnitude(   d_gx,
                                    d_gy,
                                    d_magnitude);
    thrust::transform(d_gx.begin(), d_gx.end(), d_gx.begin(), d_gx.begin(), thrust::plus<float>()); 
    thrust::transform(d_gy.begin(), d_gy.end(), d_gy.begin(), d_gy.begin(), thrust::plus<float>()); 
    thrust::copy(
               thrust::make_zip_iterator(
                                   thrust::make_tuple(d_magnitude.begin(), d_magnitude.begin(), d_magnitude.begin())),
               thrust::make_zip_iterator(
                                   thrust::make_tuple(d_magnitude.end(), d_magnitude.end(), d_magnitude.end())),
               thrust::make_zip_iterator(
                                   thrust::make_tuple(h_red.begin(), h_green.begin(), h_blue.begin()))); 

    thrust::device_vector<float> d_directions(nsize);
    ced::gpu::calculateDirections(d_gx, d_gy, d_directions);
    // -----------------------------------nonmaximumSupression----------------------------
    // max value should be corrected
    thrust::device_vector<int> d_edgePoints = ced::gpu::d_nonMaximumSupression(d_directions, d_magnitude, 0.5f, height, width, nsize);
    // -----------------------------------hysteris ---------------------------------------
    // hysteysis is not workign -> thinEdges are the white points
    thrust::device_vector<int> d_thinEdges = ced::gpu::hysterysis(d_edgePoints, d_magnitude, 0.5f, 1.0f, width);
    // -----------------------------------get white pixels--------------------------------
    typedef thrust::device_vector<float>::iterator valueItr;
    typedef thrust::device_vector<int>::iterator intItr;
    thrust::permutation_iterator<valueItr, intItr> edges_itr(d_magnitude.begin(), d_thinEdges.begin());
    int pts = d_thinEdges.size();
    std::cout<<pts<<std::endl;
    thrust::fill(d_magnitude.begin(), d_magnitude.end(), 0.0f);
    thrust::fill(edges_itr, edges_itr + pts, 1.0f);
    // -----------------------------------triangulation-----------------------------------
    // calculate all the x and corresponding ys 
    thrust::device_vector<float> d_x(pts);
    thrust::device_vector<float> d_y(pts);
    thrust::transform(
                thrust::make_zip_iterator(
                        thrust::make_tuple(d_x.begin(), d_y.begin())),
                thrust::make_zip_iterator(
                        thrust::make_tuple(d_x.end(), d_y.end())),
                thrust::make_zip_iterator(
                        thrust::make_tuple(d_x.begin(), d_y.begin())),
                convertTo2D(width));
    thrust::device_vector<int> d_triangles();
    // -----------------------------------colour the triangles---------------------------- 
    // -----------------------------------assign pixel to triangles-----------------------
    // -----------------------------------assign colour to pixels------------------------
    // -----------------------------------save image--------------------------------------
    thrust::copy(d_magnitude.begin(), d_magnitude.end(), h_red.begin());
    thrust::copy(d_magnitude.begin(), d_magnitude.end(), h_green.begin());
    thrust::copy(d_magnitude.begin(), d_magnitude.end(), h_blue.begin());

    img.setRedChannel(h_red);
    img.setGreenChannel(h_green);
    img.setBlueChannel(h_blue);
    img.setHeight(height);
    img.setWidth(width);
    img.saveImage(finalout,true);
    return 0;
}
