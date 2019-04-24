#include "Image.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "ImageApplyFilter.cuh"
#include "ConvertToGrayScale.cuh"

namespace ced
{
    namespace gpu
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
                for(unsigned int id = 0; id < m_width * m_height; ++id)
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
        void Image::applyFilter(std::vector<float> _filter, int _dimension)
        {
            // ----------------------host allocation----------------------------------
            // we have the filter 
            // we have the original image
            int nwidth = m_width - _dimension  + 1;
            int nheight = m_height - _dimension + 1;
            // --------------------device allocation----------------------------------
            thrust::device_vector<float> d_oimage = m_pixelData;
            thrust::device_vector<float> d_nimage(nheight * nwidth * m_channels);
            thrust::device_vector<float> d_filter = _filter;            
            // --------------------typecast raw ptr-----------------------------------
            float* d_oimage_ptr = thrust::raw_pointer_cast(d_oimage.data());
            float* d_nimage_ptr = thrust::raw_pointer_cast(d_nimage.data());
            float* d_filter_ptr = thrust::raw_pointer_cast(d_filter.data());
            // --------------------execution config-----------------------------------
            int blockW = 32;
            int blockH = 32;
            const dim3 grid(iDivUp(nwidth, blockW),
                            iDivUp(nheight, blockH));
            const dim3 threadBlock(blockW, blockH);
            // --------------------calling kernel-------------------------------------
            d_applyFilter<<<grid, threadBlock>>>(d_oimage_ptr, 
                                                 d_nimage_ptr,   
                                                 d_filter_ptr,
                                                 nwidth,
                                                 nheight, 
                                                 _dimension);
            cudaDeviceSynchronize();
            // --------------------back to host---------------------------------------
            std::vector<float> h_nimage(d_nimage.begin(), d_nimage.end());
            // ------------------init back to host------------------------------------
            m_pixelData.resize(nheight*nwidth*m_channels);
            m_pixelData = std::move(h_nimage);
            m_width = std::move(nwidth);
            m_height = std::move(nheight);
        }
        //----------------------------------------------------------------------------
        void Image::convertToGrayscale()
        {
            // allocate to device
            thrust::device_vector<float> d_red = m_red;
            thrust::device_vector<float> d_green = m_green;
            thrust::device_vector<float> d_blue = m_blue;
            // call function
            convertToGrayScale(d_red, d_green, d_blue);
            // copy back to host
            thrust::copy(d_red.begin(), d_red.end(), m_red.begin());
            thrust::copy(d_green.begin(), d_green.end(), m_green.begin());
            thrust::copy(d_blue.begin(), d_blue.end(), m_blue.begin());
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
        void Image::saveImage(const char* _path, bool _rgb)
        {
            using namespace OIIO;
            auto out = ImageOutput::create(_path);
            if(!out) std::cerr<< "path does not exist" << std::endl;
            else
            {
                if(_rgb)
                {
                    for(unsigned int id = 0; id < m_width * m_height; ++id)
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


