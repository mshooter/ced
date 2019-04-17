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
                m_pixelData = thrust::host_vector<float>(m_width * m_height * m_channels);
                m_r = thrust::host_vector<float>(m_width * m_height);
                m_g = thrust::host_vector<float>(m_width * m_height);
                m_b = thrust::host_vector<float>(m_width * m_height);
                in->read_image(TypeDesc::FLOAT, &m_pixelData[0]);
                for(unsigned int pix = 0; pix < m_width * m_height; ++pix)
                {
                    // might use a transform iterator
                    m_r[pix] = m_pixelData[pix * m_channels + 0];
                    m_g[pix] = m_pixelData[pix * m_channels + 1];
                    m_b[pix] = m_pixelData[pix * m_channels + 2];
                }
                in->close();
            }
        }
        //----------------------------------------------------------------------------
        void Image::setPixelData(thrust::host_vector<float> _pixels)
        {
            m_pixelData = (_pixels);
        }
        //----------------------------------------------------------------------------
        void Image::setWidth(int _width)
        {
            m_width = (_width);
        }
        //----------------------------------------------------------------------------
        void Image::setHeight(int _height)
        {
            m_height = (_height);
        }
        //----------------------------------------------------------------------------
        void Image::setChannels(int _channels)
        {
            m_channels = (_channels);
        }
        //----------------------------------------------------------------------------
        void Image::applyFilter(thrust::host_vector<float> _filter, int _dimension)
        {
            int nwidth = m_width - _dimension  + 1;
            int nheight = m_height - _dimension + 1;
            thrust::host_vector<float> nvimage (nheight*nwidth*m_channels, 0.0f);
            for(int x=0; x < nheight * nwidth; ++x)
            {
                int j = x % nwidth;
                int i = x / nwidth;

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
            m_pixelData = (nvimage);
            m_width = (nwidth);
            m_height = (nheight);
        }
        //----------------------------------------------------------------------------
        void Image::convertToGrayscale()
        {
            for(int i=0; i < m_height; ++i)
            {
                for(int j=0; j < m_width; ++j)
                {
                    float pixelData = (
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
        thrust::host_vector<float> Image::getPixelData()
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
        thrust::host_vector<float> Image::getRedChannel()
        {
            return m_r;
        } 
        //----------------------------------------------------------------------------
        thrust::host_vector<float> Image::getBlueChannel()
        {
            return m_b;
        } 
        //----------------------------------------------------------------------------
        thrust::host_vector<float> Image::getGreenChannel()
        {
            return m_g;
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


