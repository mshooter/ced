# LowPolyArt

## Introduction 
Programming project for Bournemouth university, for the unit Advanced programming. 
The goal of the project is to either take a serialized version of someone else and optimise it or to create your own serialised program and optimise it.

### My Project
#### Serialize
I try to replicate low poly art.
The process of reaching that goal was to: 
1. Implement an edge detection 
2. Generate random pixels -> triangles not only focused on the edge
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

## Results
Benchmarked on:
* GPU: NVIDIA QUADRO K2200 
* CPU: ... 

### gaussian filter 
####[dimension = 3, sigma = 0.2f, sigma = 1.4f, sigma = 2.5f]
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_gaussianFilter     |        252 ns |        252 ns|    2471912|
|CPU_gaussianFilter     |        251 ns |        251 ns|    2589179|
|CPU_gaussianFilter     |        251 ns |        251 ns|    2589734|

####[dimension = 5, sigma= 0.2f, sigma = 1.4f, sigma = 2.5f]
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_gaussianFilter     |        654 ns |      654  ns |    1021431|
|CPU_gaussianFilter     |       690 ns  |       689 ns |     988796|
|CPU_gaussianFilter     |        703ns  |       703 ns |     716667|
####[dimension = 7, sigma = 0.2f, sigma = 1.4f, sigma = 2.5f]
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_gaussianFilter     |       1205 ns |       1205 ns|     455001|
|CPU_gaussianFilter     |       1391 ns |       1391 ns|     387382|
|CPU_gaussianFilter     |       1390 ns |       1389 ns|     411988|



### Edge detection + get all the white pixels
if pixel is higher than upper maxValue the pixel is accepted as edge
if pixel is lower than lower threshold it is rejected
if inbetween - then it will be accepted only if it is connected to a pixel that is above the upper threshold 
#### 246x246 image [minvalue = 0.4f, maxvalue - 0.7f]
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_grayscale          |    332131 ns  |    331460 ns |       2090|
|CPU_applyFilter        |   9117598 ns  |   9098777 ns |         62|
|CPU_calculateGradients |   7037155 ns  |   7036687 ns |         80|
|CPU_nonMaxSup          |   1541691 ns  |   1541611 ns |        438|
|CPU_hysterysis         |    604565 ns  |    604527 ns |       1143|
|CPU_getWhitePixels     |    385196 ns  |    385173 ns |       1679|
#### 494x340 image [minvalue = 0.4f, maxvalue - 0.7f] 
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_grayscale          |   343592 ns   |   343571 ns  |      1900 |
|CPU_applyFilter        |  9376179 ns   |  9375490 ns  |        58 |
|CPU_calculateGradients |  7140735 ns   |  7140310 ns  |        79 |
|CPU_nonMaxSup          |  1467957 ns   |  1467873 ns  |       478 |
|CPU_hysterysis         |   606336 ns   |   606301 ns  |      1126 |
|CPU_getWhitePixels     |   374204 ns   |   374181 ns  |      1738 |
#### 768x768 image [minvalue = 0.4f, maxvalue - 0.7f] 
|Benchmark               |        Time    |        CPU  |Iterations|
|------------------------|----------------|-------------|----------|
|CPU_grayscale           |  1168286 ns    | 1168199 ns  |       590|
|CPU_applyFilter         | 59477550 ns    |59473960 ns  |        11|
|CPU_calculateGradients  | 32021747 ns    |32020065 ns  |        21|
|CPU_nonMaxSup           |  6195711 ns    | 6195232 ns  |       108|
|CPU_hysterysis          |  2021862 ns    | 2021735 ns  |       352|
|CPU_getWhitePixels      |  1146384 ns    | 1146312 ns  |       648|
#### 1280x720 image [minvalue = 0.4f, maxvalue - 0.7f]
|Benchmark              |         Time    |        CPU  |Iterations|
|-----------------------|-----------------|-------------|----------|
|CPU_grayscale          |   1798445 ns    | 1798350 ns  |       378|
|CPU_applyFilter        |  96182290 ns    |96176575 ns  |         7|
|CPU_calculateGradients |  49597939 ns    |49594847 ns  |        13|
|CPU_nonMaxSup          |   9555497 ns    | 9554871 ns  |        73|
|CPU_hysterysis         |   3191339 ns    | 3191145 ns  |       213|
|CPU_getWhitePixels     |   1652739 ns    | 1652559 ns  |       439|
### Tiangulation 
#### triangulation of 10 points
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_triangulateWhitePix|      3743 ns  |      3736 ns |     189593|
|CPU_assignPixToTri     |      2714 ns  |      2709 ns |     261332|
|CPU_assignColToPix     |      1552 ns  |      1549 ns |     454531|

#### triangulation of half the white pixels - 246x246: amount of pixels = 1240 
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_triangulateWhitePix|   5520473 ns  |   5520186 ns |        127|
|CPU_assignPixToTri     |   4309320 ns  |   4309072 ns |        165|
|CPU_assignColToPix     | 103549667 ns  | 103544366 ns |          7|
#### triangulation of all white the white pixels - 246x246: amount of pixels = 2481 
|Benchmark              |         Time |            CPU |  Iterations|
|-----------------------|--------------|----------------|------------|
|CPU_triangulateWhitePix|   6239150 ns |    6238687 ns  |       110  |
|CPU_assignPixToTri     |   8790107 ns |    8789450 ns  |        77  |
|CPU_assignColToPix     | 232454711 ns |  232441174 ns  |         3  |
#### triangulation of half the white pixels - 494x340: amount of pixels = 4789
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_triangulateWhitePix| 189270664 ns  | 189258991 ns |          4|
|CPU_assignPixToTri     |  27543460 ns  |  27541835 ns |         26|
|CPU_assignColToPix     |2262565676 ns  |2262443427 ns |          1|
#### triangulation of all white the white pixels - 494x340: amount of pixels = 9579
|Benchmark              |         Time    |        CPU      |Iterations| 
|-----------------------|-----------------|-----------------|----------|
|CPU_triangulateWhitePix| 577721598 ns    |   577586568 ns  |         1|
|CPU_assignPixToTri     |  56862544 ns    |    56350253 ns  |        13|
|CPU_assignColToPix     |5791823019 ns    |  5754552533 ns  |         1|
#### triangulation of half the white pixels -768x768: amount of pixels = 9576
|Benchmark              |         Time    |        CPU      |Iterations|
|-----------------------|-----------------|-----------------|----------|
|CPU_triangulateWhitePix|    405892122 ns |    405870785 ns |        2 |
|CPU_assignPixToTri     |     78483771 ns |     78479652 ns |        9 |
|CPU_assignColToPix     |   5231109502 ns |   5230820930 ns |        1 |
#### triangulation of all the white pixels -768x768: amount of pixels = 19152 
|Benchmark              |         Time    |        CPU      |Iterations|
|-----------------------|-----------------|-----------------|----------|
|CPU_triangulateWhitePix|    447707853 ns |   447665192 ns  |         2|
|CPU_assignPixToTri     |     79499049 ns |    79491342 ns  |         9|
|CPU_assignColToPix     |   1933013743 ns |  1932900934 ns  |         1|
#### triangulation of half the white pixels - 1280x720: amount of pixels = 11589
|Benchmark              |         Time    |        CPU     |Iterations|
|-----------------------|-----------------|----------------|----------|
|CPU_triangulateWhitePix|  497367690 ns   |   497326031 ns |      2   |
|CPU_assignPixToTri     |  179481345 ns   |   179470635 ns |      4   | 
|CPU_assignColToPix     |17730416212 ns   | 17729212685 ns |      1   |
#### triangulation of all the white pixels - 1280x720: amount of pixels = 231789
|Benchmark              |         Time  |          CPU | Iterations|
|-----------------------|---------------|--------------|-----------|
|CPU_triangulateWhitePix| 686308297 ns  | 686254359 ns |          1|
|CPU_assignPixToTri     | 247703107 ns  | 247690127 ns |          3|
|CPU_assignColToPix     |7623212109 ns  |7622800483 ns |          1|

