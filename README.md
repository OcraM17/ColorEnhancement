# Color Enhancement
Color Enhancement is a very famous image processing/computer vision problem. The goal of this work is to implement different color losses
in order to learn the best possible color transformation. The base architecture of this work is a CNN: as explained by Bianco et 
al [1], this CNN takes as input an image and it estimates the parameters of a global image transformation. These parameters are linear combined with
a basis function in order to produce a color transformation. This color transformation is then applied to the original image via a enhancement module.
In the original work the loss function was the DeltaE Loss between the ground truth image and the enhanced image.
We decided extend this work using the idea presented by Zhang et al in [2]. Once the enhanced image is provided by the enhanced module, 3 different 
loss could be selected. This choice of changing the loss function was done in order to explore different color space losses  and improve the performance showed in [1].

## RGB Quantization Loss
The basic idea was to divide the r, g and b channel in n levels. Once we computed this quantization we built a table 3x(levels^3) with the color intervals.
We computed the euclidean distance between the table and each pixel of the enhanced image. We repeated this computation with the gt image. Then, we converted the 
pixel distance in a distribution applying the softmax operator. Once we obtained these 2 distributions we computed the crossentropy-loss.

## LAB Quantization Loss
First of all, we converted the RGB enhanced image (and the gt-image) in the LAB color space.
For the LAB color space we decided to repeat the quantization process for the channels A and B obtaining the AB crossentropy-loss. For the L channel we computed the euclidean distance between the L channel of the enhanced image and the L channel of the gt-image. We computed the overall loss as the weighted sum of the two loss components.

### Lab Relu Softmax
To increase the value of high probable bins in the histogram, we decided to implement the Relu quantization [3]. With this choice, the high probable bin has the highr value, its nearest bins have a 'medium value' and the the farest bins have a very low value (almost 0). This increased the performance of the algorithm.

## LCH Quantization Loss
The RGB enhanced image (and gt) was converted in the LCH color space. For the L and C channels, we computed the euclidean distance between the enhanced and gt-image. Then we quantized and converted the H channel to distribution. The final H loss was computed as the crossentropy between the distribution of the H channel of the enhanced image and the distribution of the H channel of the gt-image. Then we computed the final loss as the weighted sum of the L, C and H losses.

## Gan Colorization
Once we completed these experiments we decided to implements a DC-GAN based loss function. We followed the procedure explained by Goodfellow et al in [4] and modified it in order to solve our problem. The Generator of the architecture, is the CNN used before. Once obtained the enhanced image, we applied the LAB relu Softmax separately on channel L and AB. The L and AB histograms were provided to an MLP Discriminator architecture. The job of the Discriminator is to analyze the image provided in input and classify it as real or fake. In order to increase the quality of the enhanced image, we passed the enhanced image to a resnet34[5]. The output vector of the resnet was provided with the L and the AB histograms to the discriminator.
This choice was very effective as it is possible to observe from the results.

# Results
The Dataset used is the MIT-Adobe FiveK Dataset[6]. It is composed of 5000 images in the RAW format. For each of these images five enhanced version are provided (each version of the images was retouched by one among five different experts).
Here are available the results of best approach among the implemented, i.e. the Gan colorization.(first row, second row expert, third row RAW)
<p float="center">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/enhanced/2024.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/enhanced/4061.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/enhanced/4065.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/enhanced/4074.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/enhanced/4082.png" width="160" height="160">
</p>
<p float="center">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/expert/2024.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/expert/4061.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/expert/4065.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/expert/4074.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/expert/4082.png" width="160" height="160">
</p>

<p float="center">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/raw/2024.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/raw/4061.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/raw/4065.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/raw/4074.png" width="160" height="160">
<img src="https://github.com/OcraM17/ColorEnhancement/blob/master/results/raw/4082.png" width="160" height="160">
</p>

## References
[1] Bianco, Simone, et al. "Learning parametric functions for color image enhancement." International Workshop on Computational Color Imaging. Springer, Cham, 2019. \
[2] Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European conference on computer vision. Springer, Cham, 2016. \
[3] ref \
[4] Goodfellow, Ian J., et al. "Generative adversarial networks." arXiv preprint arXiv:1406.2661 (2014). \
[5] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. \
[6] V. Bychkovsky, S. Paris, E. Chan, and F. Durand. "Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs" IEEE Computer Vision and Pattern Recognition (CVPR). June 2011, Colorado Springs, CO.
