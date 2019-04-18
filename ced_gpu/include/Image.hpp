#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

#include <thrust/device_vector.h>
#include <OpenImageIO/imageio.h>
#include <vector>

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
                void Image::setPixelData(
                                    const std::vector<float>& _red, 
                                    const std::vector<float>& _green, 
                                    const std::vector<float>& _blue);
                void setWidth(int _width);
                void setHeight(int _height);
                void setChannels(int _channels);
                void getRedChannel(std::vector<float> _red);
                void getBlueChannel(std::vector<float> _blue);
                void getGreenChannel(std::vector<float> _green);
                void applyFilter(const std::vector<float>& _filter, int _dimension);
                void convertToGrayscale();
                std::vector<float> getPixelData();
                int getWidth();
                int getHeight();
                int getChannels();
                std::vector<float> getRedChannel();
                std::vector<float> getBlueChannel();
                std::vector<float> getGreenChannel();
                void saveImage(const char* _path);
            private:
                int m_width; 
                int m_height; 
                int m_channels; 
                std::vector<float> m_red;
                std::vector<float> m_green;
                std::vector<float> m_blue;
                std::vector<float> m_pixelData;
        }; 
    }
}

#endif // IMAGE_H_INCLUDED
