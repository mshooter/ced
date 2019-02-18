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
            void setPixelData(std::vector<float> _pixels);
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
            void applyFilter(std::vector<float> _filter, int _dimension);

            std::vector<float> getPixelData();
            int getWidth();
            int getHeight();
            int getChannels();
            void saveImage(const char* _path);

        private:
            int m_width; 
            int m_height; 
            int m_channels; 
            std::vector<float> m_pixelData;
    };
}
#endif //__IMAGE_H_INCLUDED__

