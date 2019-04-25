// Image.hpp 
#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

#include <OpenImageIO/imageio.h>
#include <vector>

namespace ced 
{
    namespace cpu
    {
        class Image
        {
            public:
                Image(const char* _path); 
                Image() = default; 
                ~Image(); 
                void setPixelData(std::vector<float> _pixels);
                void setWidth(int _width);
                void setHeight(int _height);
                void setChannels(int _channels);
                void setRedChannel(const std::vector<float>& _red);
                void setGreenChannel(const std::vector<float>& _green);
                void setBlueChannel(const std::vector<float>& _blue);
                //------------------------------------------------------------
                /// @build applying a filter
                /// @param[_in] _img : input image 
                /// @param[_in] _filter : filter you are using 
                /// @param[_in] _dimension : dimension of filter 
                /// @return : return image with applied filter
                //------------------------------------------------------------
                void applyFilter(std::vector<float> _filter, int _dimension);
                void convertToGrayscale();
                std::vector<float> getPixelData();
                int getWidth();
                int getHeight();
                int getChannels();
                std::vector<float> getRedChannel();
                std::vector<float> getGreenChannel();
                std::vector<float> getBlueChannel();
                void saveImage(const char* _path, bool _rgb = false);
        
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
#endif //IMAGE_H_INCLUDED
