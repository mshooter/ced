#include "Image.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#include "ImageApplyFilter.cuh"
#include "Math.cuh"

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
            // ----------------------host allocation----------------------------------
            // we have the filter 
            // we have the original image
            int nwidth = m_width - _dimension  + 1;
            int nheight = m_height - _dimension + 1;
            // --------------------device allocation----------------------------------
            thrust::device_vector<float> d_ored     = m_red;
            thrust::device_vector<float> d_ogreen   = m_green;
            thrust::device_vector<float> d_oblue    = m_blue;

            thrust::device_vector<float> d_nred(nheight * nwidth);
            thrust::device_vector<float> d_ngreen(nheight * nwidth);
            thrust::device_vector<float> d_nblue(nheight * nwidth);

            thrust::device_vector<float> d_filter = _filter;            
            // --------------------typecast raw ptr-----------------------------------
            float* d_ored_ptr   = thrust::raw_pointer_cast(d_ored.data());
            float* d_ogreen_ptr = thrust::raw_pointer_cast(d_ogreen.data());
            float* d_oblue_ptr  = thrust::raw_pointer_cast(d_oblue.data());

            float* d_nred_ptr    = thrust::raw_pointer_cast(d_nred.data());
            float* d_ngreen_ptr  = thrust::raw_pointer_cast(d_ngreen.data());
            float* d_nblue_ptr   = thrust::raw_pointer_cast(d_nblue.data());

            float* d_filter_ptr = thrust::raw_pointer_cast(d_filter.data());
            // --------------------execution config-----------------------------------
            int blockW = 32;
            int blockH = 32;
            const dim3 grid(iDivUp(nwidth, blockW),
                            iDivUp(nheight, blockH));
            const dim3 threadBlock(blockW, blockH);
            // --------------------calling kernel-------------------------------------
            d_applyFilter<<<grid, threadBlock>>>(d_ored_ptr, 
                                                 d_ogreen_ptr,
                                                 d_oblue_ptr,
                                                 d_nred_ptr,   
                                                 d_ngreen_ptr,   
                                                 d_nblue_ptr,   
                                                 d_filter_ptr,
                                                 nwidth,
                                                 nheight, 
                                                 _dimension);
            cudaDeviceSynchronize();
            // --------------------back to host---------------------------------------
            m_red.resize(nheight*nwidth);
            m_green.resize(nheight*nwidth);
            m_blue.resize(nheight*nwidth);

            thrust::copy(d_nred.begin()     , d_nred.end()  , m_red.begin());
            thrust::copy(d_ngreen.begin()   , d_ngreen.end(), m_green.begin());
            thrust::copy(d_nblue.begin()    , d_nblue.end() , m_blue.begin());

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
            thrust::device_vector<float> d_result(m_red.size());
            // sum red and green
            thrust::transform(  d_red.begin(), 
                                d_red.end(), 
                                d_green.begin(), 
                                d_result.begin(), 
                                thrust::plus<float>());

            thrust::transform(  d_result.begin(), 
                                d_result.end(), 
                                d_blue.begin(), 
                                d_result.begin(), 
                                thrust::plus<float>());
            // DIVIDE
            thrust::transform(  d_result.begin(), 
                                d_result.end(), 
                                d_result.begin(), 
                                divideByConstant<float>(3.0f));
            // copy back to host
            thrust::copy(d_result.begin()   ,   d_result.end()  , m_red.begin());
            thrust::copy(d_result.begin()   ,   d_result.end()  , m_green.begin());
            thrust::copy(d_result.begin()   ,   d_result.end()  , m_blue.begin());
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


