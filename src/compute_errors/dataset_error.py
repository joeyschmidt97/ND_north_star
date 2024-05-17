import numpy as np
import matplotlib.pyplot as plt



def SE_2D_image(original_ND_data, predicted_ND_data):
    """
    Compute the Mean Squared Error between two images
    """
    SE_data = compute_ND_error(original_ND_data, predicted_ND_data)
    
    fig, ax = plt.subplots(figsize=(6, 6))  # Set a consistent figure size
    plt.imshow(SE_data)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Values', ticks=[0, 1])
    plt.title('2D Image Plot')

    # Set x and y limits from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Ensure equal scaling for both axes
    ax.set_aspect('equal')

    plt.show()






def compute_ND_error(original_ND_data:list, predicted_ND_data:list):
    """
    Compute the squared error between two dataset
    """
    # Check if the images have the same shape
    assert original_ND_data.shape == predicted_ND_data.shape, "Datasets must have the same shape"
    
    SE_data = (original_ND_data - predicted_ND_data)**2

    return SE_data

