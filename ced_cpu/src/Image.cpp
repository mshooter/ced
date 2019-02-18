#include "Image.hpp"

namespace ced
{
    using namespace OIIO;
    Image::Image(const char* _path)
    {
        std::unique_ptr<ImageInput> in = ImageInput::open(_path);
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
    void Image::saveImage(const char* _path, int _width, int _height, int _channels, std::vector<float> _pixData)
    {
        using namespace OIIO;
        std::unique_ptr<ImageOutput> out = ImageOutput::create(_path);
        if(!out) std::cerr<< "path does not exist" << std::endl;
        else
        {
            ImageSpec spec(_width, _height, _channels, TypeDesc::FLOAT);
            std::cout<<"successfully saved image"<<std::endl;
            out->open(_path, spec);
            out->write_image(TypeDesc::FLOAT, &_pixData[0]);
            out->close();
        } 
    }

}

