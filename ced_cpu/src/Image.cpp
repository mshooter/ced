#include "Image.hpp"

namespace ced
{
    Image::Image(const char* _path)
    {
        using namespace OIIO;
        auto in = ImageInput::open(_path);
        if(!in) std::cerr<<"Could not find image "<<std::endl;
        else
        {
            std::cout<<"Find image"<<std::endl;
            const ImageSpec &spec = in->spec();
            m_width = std::move(spec.width);
            m_height = std::move(spec.height);
            m_channels = std::move(spec.nchannels); 
            m_pixelData = std::vector<float>(m_width * m_height * m_channels);
            in->read_image(TypeDesc::FLOAT, &m_pixelData[0]);
            in->close();
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
    void Image::applyFilter(std::vector<float> _filter, int _dimension)
    {
        int nwidth = m_width - _dimension  + 1;
        int nheight = m_height - _dimension + 1;
        std::vector<float> nvimage (nheight*nwidth*m_channels, 0.0f);
        for(int i=0; i < nheight; ++i)
        {
            for(int j=0; j < nwidth; ++j)
              {
                 for(int h=i; h < i + _dimension; ++h)
                 {
                     for(int w=j; w < j + _dimension; ++w)
                     {
                          int base = (j+i*nwidth)* m_channels;
                          int ibase = (w+h*m_width) * m_channels;
                          int fbase = ((h-i) + (w-j) * _dimension);
                          nvimage[base+0] +=  m_pixelData[ibase+0] * _filter[fbase];
                          nvimage[base+1] +=  m_pixelData[ibase+1] * _filter[fbase];
                          nvimage[base+2] +=  m_pixelData[ibase+2] * _filter[fbase];
                     }
                 }
              }
          }
        m_pixelData = std::move(nvimage);
        m_width = std::move(nwidth);
        m_height = std::move(nheight);
    }
    //----------------------------------------------------------------------------
    void Image::convertToGrayscale()
    {
        for(int i=0; i < m_height; ++i)
        {
            for(int j=0; j < m_width; ++j)
            {
                float pixelData =
                (
                    m_pixelData[(j+i*m_width) * m_channels + 0]  +  
                    m_pixelData[(j+i*m_width) * m_channels + 1]  +  
                    m_pixelData[(j+i*m_width) * m_channels + 2]
                ) / 3.0f; 
                m_pixelData[(j+i*m_width) * m_channels + 0 ] = pixelData;
                m_pixelData[(j+i*m_width) * m_channels + 1 ] = pixelData;
                m_pixelData[(j+i*m_width) * m_channels + 2 ] = pixelData;
            }
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
    int Image::getChannels()
    {
        return m_channels;
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
            std::cout<<"successfully saved image"<<std::endl;
            out->open(_path, spec);
            out->write_image(TypeDesc::FLOAT, &m_pixelData[0]);
            out->close();
        } 
    }

}

