# ND_north_star
Explore higher dimensional data and determine the optimal path to get to a target point/region (north star)


# Solving the problem
This project uses 2D-Perlin noise (typically for terrain map generation in videogames) to mimic physical systems that are continuously varying. To simplify the data processing, we round the Perlin noise to 0 or 1 creating "good" or "bad" points respectively.

<!-- ![Perlin Noise rounded|200](https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise.png?raw=true) -->
<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise.png" width="512">

To create more complex data, we can increase the octaves of the Perlin noise in order to optimize performance across more complex data.

<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_octave_1.png" width="350"><img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_octave_8.png" width="350">

As can be seen in the images, the traversable pathways (white points) vary significantly between each image. Therefore, an edge detection algorithm would aid in finding the boundaries of these pathways.

<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_boundaries.png" width="512">

__However__, because each of these points can be thought of as an experiments, they may be costly or timely (or both) to obtain. Therefore, we strive to obtain the boundary given minimal amount of points using a machine learning algorithm.

<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_boundaries.png" width="350">
<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_boundaries_sparse.png" width="350">


# Dataset
Our data consist of sparse 2D Perlin noise of resolution (30,30) ranging across octaves 1-4. The number of points dropped in each image ranges from 0-95% to test for model robustness.

# Stakeholders
- Machine learning optimization engineers
- Scientists/Engineers using expensive equipment for data collection

# KPI (Key Performance Indicators)
- The MSE (mean-squared error) of the sparse image boundary compared to the actual image boundary to give an overall measure of performance across all points (black and white).
- The WCE (weighted-cross entropy) of the boundary compared between the actual image and the model-reconstructed boundary. This gives a weighted error tuned to the thickness and continuity of the boundary which the MSE cannot capture.

# Models
## Zero-Filling-CNN model:
- We fill any missing data with 0s to complete the image in order to feed it into our convolutional neural network (CNN) model
- Two convolution layers, one MaxPooling layer, then two deconvolution layers. 
- We use the Sigmoid function as our activation function in the last layer to get values between 0 and 1. 
- A deconvolutional layer is the output layer ensuring an image with the boundaries is given as the output

## KNN-CNN model:
- We use k-nearest neighbor (kNN), with the 3 nearest neighbors, to fill the missing data points and use the same CNN model as above to reconstruct the boundary given minimal information

# Performance
<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/model_performance.png" width="780">

- WCE diverges while the MSE converges: Although we lose some features in the whole picture, we still guarantee pointwise accuracy at a certain level
- Zero-filling outperforms at the top of MSE: In extreme case, zero-filling guarantee 50% accuracy when kNN is out of the threshold of good performance

