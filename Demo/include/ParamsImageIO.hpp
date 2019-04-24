#ifndef PARAMSIMAGEIO_H_INCLUDED
#define PARAMSIMAGEIO_H_INCLUDED

#ifdef __APPLE__
    const char *filename    = "/Users/moirashooter/Desktop/Cat/bob.jpg";
    const char *outgray     = "/Users/moirashooter/Desktop/Cat/graycat.jpg";
    const char *outgaussian = "/Users/moirashooter/Desktop/Cat/gaussiancat.jpg";
    const char *outgradient = "/Users/moirashooter/Desktop/Cat/nonMaximumSupressioncat.jpg";
    const char *finalout    = "/Users/moirashooter/Desktop/Cat/edge_image.jpg";
#else
    const char *filename    = "/home/s4928793/Desktop/Cat/cat.jpg";
    const char *outgray     = "/home/s4928793/Desktop/Cat/graycat.jpg";
    const char *outgaussian = "/home/s4928793/Desktop/Cat/gaussiancat.jpg";
    const char *outgradient = "/home/s4928793/Desktop/Cat/nonMaximumSupressioncat.jpg";
    const char *finalout    = "/home/s4928793/Desktop/Cat/edge_image.jpg";
#endif

#endif // PARAMSIMAGEIO_H_INCLUDED
