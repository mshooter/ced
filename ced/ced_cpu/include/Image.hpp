// Image.hpp 
#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

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
            void convertToGrayscale();
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
#endif //IMAGE_H_INCLUDED
