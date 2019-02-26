# Canny Edge Detection 

## Introduction 
Programming project for Bournemouth university, for the unit Advanced programming. 

### My Project
I implemented Canny edge detection because for many it is considered as the ultimate edge detector. 
From the edge detector you get clean, thin edges thar are well connected to nearby edges. 
The detector is a multistage edge detection algorithm, the steps are: 
1) Preprocessing: You get rid of the noise, by applying a gaussian blur 
2) Calculating Gradients: Gradient magnitues and directions are calcualte at every single point in the image 
3) Nonmaximum supression: You get rid of the pixels by giving a certain threshold 
4) Thresholding hysterysis: You grow the edges that were given from the previous step

#### Structure 
#### Dependencies 
