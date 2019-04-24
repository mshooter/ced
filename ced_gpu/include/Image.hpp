// Image.hpp 
#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

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
                void saveImage(const char* _path, bool _rgb);
        
            private:
                int m_width; 
                int m_height; 
                int m_channels; 
                std::vector<float> m_red;
                std::vector<float> m_green;
                std::vector<float> m_blue;
                std::vector<float> m_pixelData;
        };
        inline int iDivUp(const unsigned int &a, const unsigned int &b){return (a%b != 0 ? (a/b+1) : (a/b));}
    }

}
#endif //IMAGE_H_INCLUDED
