#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <OpenImageIO/imageio.h>

namespace ced
{
    namespace gpu
    {
        class Image
        {
            public:
                Image(const char* _path); 
                Image() = default; 
                ~Image(); 
                void setPixelData(thrust::host_vector<float> _pixels);
                void setWidth(int _width);
                void setHeight(int _height);
                void setChannels(int _channels);
                //------------------------------------------------------------
                /// @build applying a filter
                /// @param[_in] _img : input image 
                /// @param[_in] _filter : filter you are using 
                /// @param[_in] _dimension : dimension of filter 
                /// @return : return image with applied filter
                //------------------------------------------------------------
                void applyFilter(thrust::host_vector<float> _filter, int _dimension);
                void convertToGrayscale();
                thrust::host_vector<float> getPixelData();
                int getWidth();
                int getHeight();
                int getChannels();
                thrust::host_vector<float> getRedChannel(); 
                thrust::host_vector<float> getBlueChannel(); 
                thrust::host_vector<float> getGreenChannel(); 
                void saveImage(const char* _path);
        
            private:
                int m_width; 
                int m_height; 
                int m_channels; 
                thrust::host_vector<float> m_r;
                thrust::host_vector<float> m_g;
                thrust::host_vector<float> m_b;
                thrust::host_vector<float> m_pixelData;
        }; 
    }
}

#endif // IMAGE_H_INCLUDED
