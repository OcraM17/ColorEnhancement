# ColorEnhancement
Color Enhancement is a very famous image processing/computer vision problem. The goal of this work is to implement different color losses
in order to learn the best possible color transformation. The base architecture of this work is a CNN: as explained by Bianco et 
al [1], this CNN takes as input an image and it estimates the parameters of a global image transformation. These parameters are linear combined with
a basis function in order to produce a color transformation. This color transformation is then applied to the original image via a enhancement module.
In the original work the loss function was the DeltaE Loss between the ground truth image and the enhanced image.
We decided extend this work using the idea presented by Zhang et al in [2]. Once the enhanced image is provided by the enhanced module 3 different 
loss could be selected.

## RGB Quantization Loss
The basic idea was to divide the r, g and b channel n levels. Once we computed this quantization we built a table 3x(levels^3) with the colors intervals.
We computed the euclidean distance between the table and each pixel of the enhanced image. We repeated this distance with the gt image. Then we converted the 
pixel quantization in a distribution applying the softmax operator. Once we obtained these 2 distributions we computed the crossentropy-loss.

## LAB Quantization Loss



## References
[1] Bianco, Simone, et al. "Learning parametric functions for color image enhancement." International Workshop on Computational Color Imaging. Springer, Cham, 2019. \
[2] Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European conference on computer vision. Springer, Cham, 2016. \
[3] AAA
