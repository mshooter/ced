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

### 246x246 - 10 pixels
|Benchmark              |         Time  |          CPU | Iterations
|-----------------------|---------------|--------------|-----------
|CPU_gaussianFilter     |       690 ns  |       689 ns |     988796
|-------------------------------------------------------------------
|CPU_grayscale          |    332131 ns  |    331460 ns |       2090
|CPU_applyFilter        |   9117598 ns  |   9098777 ns |         62
|CPU_calculateGradients |   7037155 ns  |   7036687 ns |         80
|CPU_nonMaxSup          |   1541691 ns  |   1541611 ns |        438
|CPU_hysterysis         |    604565 ns  |    604527 ns |       1143
|CPU_getWhitePixels     |    385196 ns  |    385173 ns |       1679
|-----------------------|---------------|--------------|-----------
|CPU_triangulateWhitePix|      5268 ns  |      5268 ns |     133630
|CPU_assignPixToTri     |     42060 ns  |     42058 ns |      16770
|CPU_assignColToPix     |      9051 ns  |      9051 ns |      76950
### 246x246 - half the white pixels
|Benchmark              |         Time  |          CPU | Iterations
|-----------------------|---------------|--------------|-----------
|CPU_triangulateWhitePix|   5520473 ns  |   5520186 ns |        127
|CPU_assignPixToTri     |   4309320 ns  |   4309072 ns |        165
|CPU_assignColToPix     | 103549667 ns  | 103544366 ns |          7
### 246x246 - all the pixels

|Benchmark                       Time             CPU   Iterations
|-----------------------|--------------|----------------|-----------
|CPU_gaussianFilter     |       688 ns |        688 ns  |    981740
|-------------------------------------------------------------------
|CPU_grayscale          |    118083 ns |     118076 ns  |      5884
|CPU_applyFilter        |  23167242 ns |   23165824 ns  |       283
|CPU_calculateGradients |   2196203 ns |    2196076 ns  |       289
|CPU_nonMaxSup          |    519840 ns |     519810 ns  |      1351
|CPU_hysterysis         |    194025 ns |     194015 ns  |      3658
|CPU_getWhitePixels     |   7181443 ns |    7180994 ns  |       232
|CPU_triangulateWhitePix|   6239150 ns |    6238687 ns  |       110
|CPU_assignPixToTri     |   8790107 ns |    8789450 ns  |        77
|CPU_assignColToPix     | 232454711 ns |  232441174 ns  |         3

### 494x340 - 10 pixels
|Benchmark              |         Time  |          CPU | Iterations
|-----------------------|---------------|--------------|-----------
|CPU_gaussianFilter     |      704 ns   |      704 ns  |    739576
|-------------------------------------------------------------------
|CPU_grayscale          |   343592 ns   |   343571 ns  |      1900
|CPU_applyFilter        |  9376179 ns   |  9375490 ns  |        58
|CPU_calculateGradients |  7140735 ns   |  7140310 ns  |        79
|CPU_nonMaxSup          |  1467957 ns   |  1467873 ns  |       478
|CPU_hysterysis         |   606336 ns   |   606301 ns  |      1126
|CPU_getWhitePixels     |   374204 ns   |   374181 ns  |      1738
|-----------------------|-----------------------------------------    
|CPU_triangulateWhitePix|     5138 ns   |     5137 ns  |    137716
|CPU_assignPixToTri     |    40548 ns   |    40545 ns  |     17008
|CPU_assignColToPix     |     8535 ns   |     8534 ns  |     80130
### 494x340 - half the white pixels
|Benchmark              |         Time  |          CPU | Iterations
|-----------------------|---------------|--------------|-----------
|CPU_triangulateWhitePix| 189270664 ns  | 189258991 ns |          4
|CPU_assignPixToTri     |  27543460 ns  |  27541835 ns |         26
|CPU_assignColToPix     |2262565676 ns  |2262443427 ns |          1
### 494x340 - all the pixels

|Benchmark              |         Time    |        CPU      |Iterations
|-----------------------|-----------------|-----------------|----------
|CPU_gaussianFilter     |       700 ns    |         700 ns  |    731665
|-------------------------------------------------------------------
|CPU_grayscale          |    325228 ns    |      325206 ns  |      2028
|CPU_applyFilter        |  13082935 ns    |    13081590 ns  |        44
|CPU_calculateGradients |   7276140 ns    |     7275482 ns  |        76
|CPU_nonMaxSup          |   1456049 ns    |     1455935 ns  |       482
|CPU_hysterysis         |    586662 ns    |      586626 ns  |      1216
|CPU_getWhitePixels     |  12112365 ns    |    12111251 ns  |       100
|CPU_triangulateWhitePix| 577721598 ns    |   577586568 ns  |         1
|CPU_assignPixToTri     |  56862544 ns    |    56350253 ns  |        13
|CPU_assignColToPix     |5791823019 ns    |  5754552533 ns  |         1

### 768x768 - 10 pixels
|Benchmark               |        Time    |        CPU  |Iterations
|------------------------|----------------|-------------|-----------
|CPU_gaussianFilter      |      685 ns    |     685 ns  |    994124
|-------------------------------------------------------------------
|CPU_grayscale           |  1168286 ns    | 1168199 ns  |       590
|CPU_applyFilter         | 59477550 ns    |59473960 ns  |        11
|CPU_calculateGradients  | 32021747 ns    |32020065 ns  |        21
|CPU_nonMaxSup           |  6195711 ns    | 6195232 ns  |       108
|CPU_hysterysis          |  2021862 ns    | 2021735 ns  |       352
|CPU_getWhitePixels      |  1146384 ns    | 1146312 ns  |       648
|-------------------------------------------------------------------
|CPU_triangulateWhitePix |     5184 ns    |    5174 ns  |    138293
|CPU_assignPixToTri      |  3908587 ns    | 3908353 ns  |       179
|CPU_assignColToPix      |  1077777 ns    | 1077006 ns  |       666
### 768x768 - half the white pixels
|Benchmark              |         Time    |        CPU      |Iterations
|-----------------------|-----------------|-----------------|----------
|CPU_triangulateWhitePix|    405892122 ns |    405870785 ns |        2
|CPU_assignPixToTri     |     78483771 ns |     78479652 ns |        9
|CPU_assignColToPix     |   5231109502 ns |   5230820930 ns |        1
### 768x768 - all the white pixels

|Benchmark              |         Time    |        CPU      |Iterations
|-----------------------|-----------------|-----------------|----------
|CPU_gaussianFilter     |          690 ns |         690 ns  |    981860
|-------------------------------------------------------------------
|CPU_grayscale          |      1185092 ns |     1185001 ns  |       592
|CPU_applyFilter        |     60563387 ns |    60556070 ns  |        11
|CPU_calculateGradients |     32420633 ns |    32418726 ns  |        22
|CPU_nonMaxSup          |      6210326 ns |     6209934 ns  |       106
|CPU_hysterysis         |      1997498 ns |     1997393 ns  |       349
|CPU_getWhitePixels     |     25926291 ns |    25924412 ns  |       100
|CPU_triangulateWhitePix|    447707853 ns |   447665192 ns  |         2
|CPU_assignPixToTri     |     79499049 ns |    79491342 ns  |         9
|CPU_assignColToPix     |   1933013743 ns |  1932900934 ns  |         1

### 1280x720 - 10 pixels
|Benchmark              |         Time    |        CPU  |Iterations
|-----------------------|-----------------|-------------|-----------
|CPU_gaussianFilter     |       676 ns    |     676 ns  |    984151
|-------------------------------------------------------------------
|CPU_grayscale          |   1798445 ns    | 1798350 ns  |       378
|CPU_applyFilter        |  96182290 ns    |96176575 ns  |         7
|CPU_calculateGradients |  49597939 ns    |49594847 ns  |        13
|CPU_nonMaxSup          |   9555497 ns    | 9554871 ns  |        73
|CPU_hysterysis         |   3191339 ns    | 3191145 ns  |       213
|CPU_getWhitePixels     |   1652739 ns    | 1652559 ns  |       439
|-------------------------------------------------------------------
|CPU_triangulateWhitePix|      5868 ns    |    5867 ns  |    124246
|CPU_assignPixToTri     |   3450834 ns    | 3450631 ns  |       205
|CPU_assignColToPix     |    136811 ns    |  136803 ns  |      5200
### 1280x720 - half the white pixels
|Benchmark              |         Time    |        CPU     |Iterations
|-----------------------|-----------------|----------------|--------
|CPU_triangulateWhitePix|  497367690 ns   |   497326031 ns |      2
|CPU_assignPixToTri     |  179481345 ns   |   179470635 ns |      4
|CPU_assignColToPix     |17730416212 ns   | 17729212685 ns |      1
### 1280x720 - all the white pixels

|Benchmark              |         Time  |          CPU | Iterations
|-----------------------|---------------|--------------|------------
|CPU_gaussianFilter     |       670 ns  |       670 ns |    1000804
|CPU_grayscale          |   1834431 ns  |   1830658 ns |        376
|CPU_applyFilter        |  95601307 ns  |  95414382 ns |          7
|CPU_calculateGradients |  49454343 ns  |  49353387 ns |         14
|CPU_nonMaxSup          |   8813598 ns  |   8791470 ns |         74
|CPU_hysterysis         |   3211050 ns  |   3210762 ns |        202
|CPU_getWhitePixels     |  32379852 ns  |  32378154 ns |        100
|CPU_triangulateWhitePix| 686308297 ns  | 686254359 ns |          1
|CPU_assignPixToTri     | 247703107 ns  | 247690127 ns |          3
|CPU_assignColToPix     |7623212109 ns  |7622800483 ns |          1

