#include <iostream>
#include <OpenImageIO/imageio.h>
#include <cmath>

int main()
{
    // read image 
    using namespace OIIO; 
    ImageInput *in = ImageInput::open("/Users/moirashooter/Desktop/cat.jpg");
    ImageOutput *out = ImageOutput::create("/Users/moirashooter/Desktop/newCat.jpg");
    if(!in)
    {
        std::cerr<<"Could not find image "<< std::endl;
    }
    else
    {
        std::cout<<"Find image"<<std::endl;
        const ImageSpec &spec = in->spec();
//        spec.set_format(TypeDesc::FLOAT);
        int width = spec.width;
        int height = spec.height;
        int channels = spec.nchannels; 
        std::vector<float> pixels(width * height * channels);
        // create filter gaussian blur
        std::vector<float> filter(5*5);
        int middle = 5/2;
        float sum = 0.0f; 
        float sigma2 = 4; 
        auto g = [&](auto x, auto y)
        {
            return std::exp(-((x*x + y*y)/(2.0f*sigma2))) / (2.0f * std::acos(-1) *sigma2);
        };
        for(int y=0; y < 5; ++y)
        {
            for(int x=0; x < 5; ++x)
            {
                filter[x+y*5] = g(x-middle, y-middle);
                sum += filter[x+y*5];
            }
        }

        for(int y=0; y < 5; ++y)
        {
            for(int x=0; x < 5; ++x)
            {
                filter[x+y*5] /= sum;
            }
        }
        // read image
        in->read_image(TypeDesc::FLOAT, &pixels[0]);
        int nwidth = width - 5 + 1;
        int nheight = height - 5 + 1;
        std::vector<float> nimage (nheight*nwidth*channels, 0.0f);
        // manipulate
        // GAUSSIAN BLUR
        for(int i=0; i < nheight; ++i)
        {
            for(int j=0; j < nwidth; ++j)
            {
                
                for(int h=i; h < i + 5; ++h)
                {
                    for(int w=j; w < j + 5; ++w)
                    {
                         int base = (j+i*nwidth)* channels;
                         int ibase = (w+h*(width)) * channels;
                         int fbase = ((h-i) + (w-j) * 5);
                         nimage[base+0] +=  pixels[ibase+0] * filter[fbase];
                         nimage[base+1] +=  pixels[ibase+1] * filter[fbase];
                         nimage[base+2] +=  pixels[ibase+2] * filter[fbase];
                    }
                }
            }
        }
        // sobel edge detector 
        std::vector<int> kernelX = {-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1};
        std::vector<int> kernelY = {-1, -2, -1,
                                     0,  0,  0,
                                    -1, -2, -1};

        for(auto& pix : nimage)
        {
            // normalize
            std::cout<<pix*255<<std::endl;
        }

        // write image
        if(!out)
        {
            std::cerr<<"path does not exist"<<std::endl;
        }
        else
        {
            ImageSpec specs(nwidth, nheight, channels, TypeDesc::FLOAT);
            std::cout<<"save"<<std::endl;
            out->open("/Users/moirashooter/Desktop/newCat.jpg", specs);
            out->write_image(TypeDesc::FLOAT, &nimage[0]);
            out->close();
            ImageOutput::destroy(out);
        }
        in->close();
        ImageInput::destroy(in);


    }
    return 0;
}
