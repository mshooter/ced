// Image.hpp 
#ifndef __IMAGE_H_INCLUDED__
#define __IMAGE_H_INCLUDED__

#include <OpenImageIO/imageio.h>
#include <vector>

namespace ced 
{
    class Image
    {
        public:
            Image(const char* _path); 
            Image() = default; 
            ~Image() = default; 
            std::vector<float> getPixelData();
            int getWidth();
            int getHeight();
            int getChannels();
            void saveImage(const char* _path,
                          int _width,
                          int _height, 
                          int _channels,
                          std::vector<float> _pixData);
        private:
            int m_width; 
            int m_height; 
            int m_channels; 
            std::vector<float> m_pixelData;
    };
}
#endif //__IMAGE_H_INCLUDED__

