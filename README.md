# Color Enhancement
Color Enhancement is a very famous image processing/computer vision problem. The goal of this work is to implement different color losses
in order to learn the best possible color transformation. The base architecture of this work is a CNN: as explained by Bianco et 
al [1], this CNN takes as input an image and it estimates the parameters of a global image transformation. These parameters are linear combined with
a basis function in order to produce a color transformation. This color transformation is then applied to the original image via a enhancement module.
In the original work the loss function was the DeltaE Loss between the ground truth image and the enhanced image.
We decided extend this work using the idea presented by Zhang et al in [2]. Once the enhanced image is provided by the enhanced module 3 different 
loss could be selected. This choice of change the loss function was done in order to explore different loss color space and improve the performance showed in [1].

## RGB Quantization Loss
The basic idea was to divide the r, g and b channel n levels. Once we computed this quantization we built a table 3x(levels^3) with the colors intervals.
We computed the euclidean distance between the table and each pixel of the enhanced image. We repeated this distance with the gt image. Then we converted the 
pixel quantization in a distribution applying the softmax operator. Once we obtained these 2 distributions we computed the crossentropy-loss.

## LAB Quantization Loss
First of all, we converted the RGB enhanced image (and the gt-image) in the LAB color space.
For the LAB color space we decide to repeate the quantization process for the channels A and B. We obtained the crossentropy-loss for the AB channels. For the L channel we computed the euclidean distance between the L channel of the enhanced image and the L channel of the gt-image. We computed the overall loss as the weighted sum of the two loss components.

## LCH Quantization Loss
The RGB image enhanced image (and gt) was converted in the LCH color space. For the L and C channels, we computed the euclidean distance between the enhanced and gt-image. Then we quantized and converted the H channel to distribution. The final H loss was computed as the crossentropy between the distribution of the H channel of the enhanced image and the distribution of the H channel of the gt-image. Then we computed the final loss as the weighted sum of the L, C and H losses.



## References
[1] Bianco, Simone, et al. "Learning parametric functions for color image enhancement." International Workshop on Computational Color Imaging. Springer, Cham, 2019. \
[2] Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European conference on computer vision. Springer, Cham, 2016. \
[3] AAA
