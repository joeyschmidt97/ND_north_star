from scipy import ndimage
from scipy.signal import convolve2d
import numpy as np


def scipy_edge_detection(data):
    # Apply a Sobel filter
    edge_sobel = ndimage.sobel(data)
    
    # You might want to threshold the edges
    edges = edge_sobel > 0.7  # Adjust threshold as needed
    
    return edges



def convolve_edge_detection(data):
    
    # Define a Sobel operator (example for 2D)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # Apply convolution
    gx = convolve2d(data, sobel_x, mode='same', boundary='wrap')
    
    # Detect edges based on gradient magnitude
    edges = np.abs(gx) > 0.5  # Adjust threshold as needed
    
    return edges
