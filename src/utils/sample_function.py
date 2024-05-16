import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

def sample_coords_and_values(coords, values, num_samples, method='random'):
    """
    Sample coordinates and values using different sampling methods.

    Parameters:
    coords (list): A list of arrays [x_arrays, y_arrays, ...] for each dimension of coordinates.
    values (array): An array of values associated with the coordinates.
    num_samples (int): The number of samples to be drawn.
    method (str): The sampling method to be used ('random' or 'lhs').

    Returns:
    sampled_coords (list): A list of sampled coordinates [sampled_x, sampled_y, ...].
    sampled_values (array): An array of values associated with the sampled coordinates.
    """
    
    num_dimensions = len(coords)
    assert all(len(arr) == len(values) for arr in coords), "Length of all coordinate arrays and values must be the same"
    
    coords_array = np.array(coords).T  # Transpose to shape (num_points, num_dimensions)

    if method == 'random':
        indices = np.random.choice(len(coords[0]), num_samples, replace=False)
    elif method == 'lhs':
        sampler = qmc.LatinHypercube(d=num_dimensions)
        sample = sampler.random(n=num_samples)
        bounds = np.array([[np.min(arr), np.max(arr)] for arr in coords])
        sampled_coords = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        
        tree = cKDTree(coords_array)
        _, indices = tree.query(sampled_coords)
    else:
        raise ValueError("Invalid sampling method. Choose 'random' or 'lhs'.")

    sampled_coords = [coords[i][indices] for i in range(num_dimensions)]
    sampled_values = values[indices]

    return sampled_coords, sampled_values
