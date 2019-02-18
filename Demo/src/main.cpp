#include <iostream>
#include <OpenImageIO/imageio.h>
#include <cmath>
#include "Image.hpp"

int main()
{
    // read image 
    using namespace OIIO; 
    // std::unique_ptr<ImageInput>
    // std::unique_ptr<ImageOutput>
    #ifdef __APPLE__
        const char *filename = "/Users/moirashooter/Desktop/cat.jpg";
        const char *outfile = "/Users/moirashooter/Desktop/ncat.jpg";
    #else
        const char *filename = "/home/s4928793/Desktop/cat.jpg";
        const char *outfile = "/home/s4928793/Desktop/bobcat.jpg";
    #endif
    ced::Image img(filename);
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
       int nwidth = img.getWidth() - 5 + 1;
       int nheight = img.getHeight() - 5 + 1;
       int channels = img.getChannels();
       std::vector<float> nimage (nheight*nwidth*channels, 0.0f);
       // manipulate
       // GAUSSIAN BLUR
       for(int i=0; i < img.getHeight(); ++i)
       {
           for(int j=0; j < img.getWidth(); ++j)
           {
              int base = (j+i*nwidth) * channels; 
              int tbase = (j+i*img.getWidth()) * channels;
              nimage[base + 0] = img.getPixelData()[tbase+0];
              nimage[base + 1] = img.getPixelData()[tbase+1];
              nimage[base + 2] = img.getPixelData()[tbase+2];
              //for(int h=i; h < i + 5; ++h)
              //{
              //    for(int w=j; w < j + 5; ++w)
              //    {
              //         int base = (j+i*nwidth)* channels;
              //         int ibase = (w+h*(img.getWidth())) * channels;
              //         int fbase = ((h-i) + (w-j) * 5);
              //         std::vector<float> pixels = std::move(img.getPixelData());
              //         nimage[base+0] +=  pixels[ibase+0] * filter[fbase];
              //         nimage[base+1] +=  pixels[ibase+1] * filter[fbase];
              //         nimage[base+2] +=  pixels[ibase+2] * filter[fbase];
              //    }
              //}
            std::cout<<i<<" "<<j<<std::endl;
           }
       }
//      // sobel edge detector 
//      std::vector<int> kernelX = {-1, 0, 1,
//                                  -2, 0, 2,
//                                  -1, 0, 1};
//      std::vector<int> kernelY = {-1, -2, -1,
//                                   0,  0,  0,
//                                  -1, -2, -1};
//

        // write image
        img.saveImage(outfile, nwidth, nheight, channels, nimage);
        
            //ImageOutput::destroy(out);
        //ImageInput::destroy(in);


    
    return 0;
}
