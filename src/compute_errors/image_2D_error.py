import numpy as np




def image_2D_error(original_image, noisy_image, error_type="MSE"):
    """
    Compute the error between two images and plot the results
    """
    # Check if the images have the same shape
    assert original_image.shape == noisy_image.shape, "Images must have the same shape"
    
    SE_image = (original_image - noisy_image) ** 2

    return error_image




def MSE_2D_image(original_image, noisy_image):
    """
    Compute the Mean Squared Error between two images
    """
    # Check if the images have the same shape
    assert original_image.shape == noisy_image.shape, "Images must have the same shape"
    
    # Compute the MSE
    mse = np.mean((original_image - noisy_image) ** 2)
    
    return mse