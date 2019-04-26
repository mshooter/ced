#include "Image.hpp"

namespace ced
{
    namespace cpu
    {
        //----------------------------------------------------------------------------
        Image::~Image()
        {
            m_pixelData.clear();
            m_red.clear();
            m_green.clear();
            m_blue.clear();
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
                m_width = spec.width;
                m_height = spec.height;
                m_channels = spec.nchannels; 
                m_pixelData = std::vector<float>(m_width * m_height * m_channels);
                in->read_image(TypeDesc::FLOAT, &m_pixelData[0]);
                in->close();
                m_red.resize(m_height*m_width);
                m_green.resize(m_height*m_width);
                m_blue.resize(m_height*m_width);
                for(int id = 0; id < m_width * m_height; ++id)
                {
                    m_red[id]     = m_pixelData[id * 3 + 0];
                    m_green[id]   = m_pixelData[id * 3 + 1];
                    m_blue[id]    = m_pixelData[id * 3 + 2];
                }
            }
        }
        //----------------------------------------------------------------------------
        void Image::setPixelData(std::vector<float> _pixels)
        {
            m_pixelData = std::move(_pixels);
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
        void Image::setRedChannel(const std::vector<float>& _red)
        {
            m_red = _red;
        }
        //----------------------------------------------------------------------------
        void Image::setGreenChannel(const std::vector<float>& _green)
        {
            m_green = _green;
        }
        //----------------------------------------------------------------------------
        void Image::setBlueChannel(const std::vector<float>& _blue)
        {
            m_blue = _blue;
        }
        //----------------------------------------------------------------------------
        void Image::applyFilter(std::vector<float> _filter, int _dimension)
        {
            int nwidth = m_width - _dimension  + 1;
            int nheight = m_height - _dimension + 1;
            std::vector<float> red  (nheight*nwidth, 0.0f);
            std::vector<float> green(nheight*nwidth, 0.0f);
            std::vector<float> blue (nheight*nwidth, 0.0f);

            for(int x=0; x < nheight * nwidth; ++x)
            {
                int i = x / nwidth;
                int j = (x % nwidth);
                for(int y = 0; y < _dimension * _dimension; ++y)
                {
                    int h = y / _dimension;
                    int w = (y %_dimension);
                    // red green blue
                    int base = x;
                    int ibase = ((w+j) + (h+i) * m_width);
                    int fbase = y;
                    red[base]   +=  m_red[ibase]    * _filter[fbase];
                    green[base] +=  m_green[ibase]  * _filter[fbase];
                    blue[base]  +=  m_blue[ibase]   * _filter[fbase];
                }
            }
            m_red.resize(nheight*nwidth);
            m_green.resize(nheight*nwidth);
            m_blue.resize(nheight*nwidth);

            m_red = std::move(red);
            m_green = std::move(green);
            m_blue = std::move(blue);
            m_width = std::move(nwidth);
            m_height = std::move(nheight);
        }
        //----------------------------------------------------------------------------
        void Image::convertToGrayscale()
        {
            for(int id = 0; id < m_height * m_width; ++id)
            {
                float pixelData = (
                    m_red[id]  +  
                    m_green[id]  +  
                    m_blue[id]
                ) / 3.0f;
                m_red[id] = pixelData;
                m_green[id] = pixelData;
                m_blue[id] = pixelData;
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
        std::vector<float> Image::getGreenChannel()
        {
            return m_green;
        }
        //----------------------------------------------------------------------------
        std::vector<float> Image::getBlueChannel()
        {
            return m_blue;
        }
        //----------------------------------------------------------------------------
        void Image::saveImage(const char* _path, bool _rgb)
        {
            using namespace OIIO;
            auto out = ImageOutput::create(_path);
            if(!out) std::cerr<< "path does not exist" << std::endl;
            else
            {
                if(_rgb)
                {
                    for(int id = 0; id < m_width * m_height; ++id)
                    {
                        m_pixelData[id * 3 + 0] = m_red[id]  ; 
                        m_pixelData[id * 3 + 1] = m_green[id]; 
                        m_pixelData[id * 3 + 2] = m_blue[id] ; 
                    }
                }
                ImageSpec spec(m_width, m_height, m_channels, TypeDesc::FLOAT);
                out->open(_path, spec);
                out->write_image(TypeDesc::FLOAT, &m_pixelData[0]);
                out->close();
            } 
        }
        //----------------------------------------------------------------------------

    }

}

