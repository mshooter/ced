# Canny Edge Detection 

## Introduction 
Programming project for Bournemouth university, for the unit Advanced programming. 
The goal of the project is to either take a serialized version of someone else and optimise it or to create your own serialised program and optimise it.

### My Project
#### Serialize
I try to replicate low poly art.
The process of reaching that goal was to: 
1. Implement an edge detection 
2. Generate random points -> triangles not only focused on the edge
3. Triangulate (delaunay)
4. Colour every pixel depending in which triangle it lies in [take the average colour of all the pixels that are inside the triangle and set the avg colour to all the pixels that lie inside that triangle] 
#### Edge Detector
I implemented Canny edge detection because for many it is considered as the ultimate edge detector. 
From the edge detector you get clean, thin edges thar are well connected to nearby edges. 
The detector is a multistage edge detection algorithm, the steps are: 
1. Preprocessing: You get rid of the noise, by applying a gaussian blur 
2. Calculating Gradients: Gradient magnitues and directions are calcualte at every single point in the image 
3. Nonmaximum supression: You get rid of the pixels by giving a certain threshold 
4. Thresholding hysterysis: You grow the edges that were given from the previous step
#### Delaunay Triangulation
Reference https://github.com/delfrrr/delaunator-cpp
#### Set colour to every pixel

