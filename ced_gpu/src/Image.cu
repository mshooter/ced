#include "Image.hpp"

namespace ced
{
    namespace gpu
    {
        //----------------------------------------------------------------------------
        Image::~Image()
        {
            m_pixelData.clear();
        }
        //----------------------------------------------------------------------------
        Image::Image(const char* _path)
        {
            using namespace OIIO;
            auto in = ImageInput::open(_path);
            if(!in) std::cerr<<"Could not find image "<<std::endl;
            else
            {
                const ImageSpec &spec = in->spec();
                m_width = (spec.width);
                m_height = (spec.height);
                m_channels = (spec.nchannels); 
                m_pixelData.resize(m_width * m_height * m_channels);
                in->read_image(TypeDesc::FLOAT, &m_pixelData[0]);
                in->close();
                m_red.resize(m_height * m_width);
                m_green.resize(m_height * m_width);
                m_blue.resize(m_height * m_width);
                for(unsigned int id = 0; id < m_height * m_width; ++id)
                {
                    m_red[id] = m_pixelData[id * m_channels + 0];
                    m_green[id] = m_pixelData[id * m_channels + 1];
                    m_blue[id] = m_pixelData[id * m_channels + 2];
                }
            }
        }
        //----------------------------------------------------------------------------
        void Image::setPixelData(const std::vector<float>& _red, 
                                 const std::vector<float>& _green, 
                                 const std::vector<float>& _blue)
        {
            for(unsigned int id = 0; id < m_height * m_widht; ++id)
            {
                m_pixelData[id * m_channels + 0] = _red[id];
                m_pixelData[id * m_channels + 1] = _green[id];
                m_pixelData[id * m_channels + 2] = _blue[id];
            }
        }
        //----------------------------------------------------------------------------
        void Image::setWidth(int _width)
        {
            m_width = std::move(_width);
        }
        //----------------------------------------------------------------------------
        void Image::setHeight(int _height)
        {
            m_height = std::move(_height);
        }
        //----------------------------------------------------------------------------
        void Image::setChannels(int _channels)
        {
            m_channels = std::move(_channels);
        }
        //----------------------------------------------------------------------------
        void Image::applyFilter(const vector<float>& _filter, int _dimension)
        {
            int nwidth = m_width - _dimension  + 1;
            int nheight = m_height - _dimension + 1;
            std::vector<float> red (nheight*nwidth, 0.0f);
            std::vector<float> green (nheight*nwidth, 0.0f);
            std::vector<float> blue (nheight*nwidth, 0.0f);
            for(int x=0; x < nheight * nwidth; ++x)
            {
                int j = x % nwidth;
                int i = x / nwidth;
                for(int h=i; h < i + _dimension; ++h)
                {
                    for(int w=j; w < j + _dimension; ++w)
                    {
                       int base     = (j+i*nwidth);
                       int ibase    = (w+h*m_width);
                       int fbase    = ((h-i) + (w-j) * _dimension);
                       red[base]    +=  red[ibase] * _filter[fbase];
                       green[base]  +=  green[ibase] * _filter[fbase];
                       blue[base]   +=  blue[ibase] * _filter[fbase];

                    }
                }
            }
            m_red = std::move(red);
            m_green = std::move(green);
            m_blue= std::move(blue);
            m_width = std::move(nwidth);
            m_height = std::move(nheight);
        }
        //----------------------------------------------------------------------------
        void Image::convertToGrayscale()
        {
            for(int id = 0; id < m_height * m_width; ++id)
            {
                float pixelData = (red[id]  +  green[id]  +  blue[id]) / 3.0f; 
                m_red[id * m_channels + 0 ]     = pixelData;
                m_green[id * m_channels + 1 ]   = pixelData;
                m_blue[id * m_channels + 2 ]    = pixelData;
            }
        }
        //----------------------------------------------------------------------------
        std::vector<float> Image::getPixelData()
        {
            return m_pixelData;
        }
        //----------------------------------------------------------------------------
        int Image::getWidth()
        {
            return m_width;
        }
        //----------------------------------------------------------------------------
        int Image::getHeight()
        {
            return m_height;
        }
        //----------------------------------------------------------------------------
        int Image::getChannels()
        {
            return m_channels;
        }
        //----------------------------------------------------------------------------
        std::vector<float> Image::getRedChannel()
        {
            return m_red;
        }
        //----------------------------------------------------------------------------
        std::vector<float> Image::getBlueChannel()  
        {
            return m_blue;
        }
        //----------------------------------------------------------------------------
        std::vector<float> Image::getGreenChannel()
        {
            return m_green;
        }
        //----------------------------------------------------------------------------
        void Image::saveImage(const char* _path)
        {
            using namespace OIIO;
            auto out = ImageOutput::create(_path);
            if(!out) std::cerr<< "path does not exist" << std::endl;
            else
            {
                ImageSpec spec(m_width, m_height, m_channels, TypeDesc::FLOAT);
                out->open(_path, spec);
                out->write_image(TypeDesc::FLOAT, &m_pixelData[0]);
                out->close();
            } 
        }
        //----------------------------------------------------------------------------

    }

}


