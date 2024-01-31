import numpy as np
import matplotlib.pyplot as plt

from perlin_noise import PerlinNoise
from mpl_toolkits.mplot3d import Axes3D


def generate_ND_perlin_noise(dimension_resolution:list, octaves:int, noise_rescaling:list=[0, 1], noise_cutoff_list:list=None):
    """
    Generate N-dimensional Perlin noise.

    Args:
    - dimension_resolution (list of int): The resolution along each dimension ([40,40] will generate 2D grid of spatial resolution 40 pixels).
    - octaves (int): The number of octaves to generate the noise (higher -> more jagged).
    - noise_rescaling (list of float, optional): Rescaling range for the generated noise between [a,b]. Default is [0, 1].
    - noise_cutoff_list (list of float, optional): Apply rounding cutoffs to the generated noise.
      The list should contain [noise_divide, noise_bottom, noise_top]. Default is None.

    Returns:
    - float or list: A single float if the function is called with a single dimension resolution,
      otherwise a nested list of floats representing the N-dimensional noise field.

    """

    # Create a PerlinNoise object with the specified number of octaves
    noise = PerlinNoise(octaves=octaves)

    noise_max = noise_rescaling[1]
    noise_min = noise_rescaling[0]

    def recursive_build(dim_index, coords):
        if dim_index == len(dimension_resolution):

            # Generate Perlin noise and rescale it to the specified range
            rescaled_noise = noise_min + (noise(coords) + 1) * (noise_max - noise_min) / 2

            # Apply noise cutoffs if provided
            if noise_cutoff_list is not None:
                noise_divide = noise_cutoff_list[0]

                # Check if noise_divide is within the range [noise_min, noise_max]
                if not (noise_min <= noise_divide <= noise_max):
                    raise ValueError(f"noise_divide ({noise_divide}) must be within the noise rescaling range [{noise_min}, {noise_max}]")

                noise_bottom = noise_cutoff_list[1]
                noise_top = noise_cutoff_list[2]

                if rescaled_noise < noise_divide:
                    rescaled_noise = noise_bottom
                else:
                    rescaled_noise = noise_top

            return rescaled_noise
        
        return [recursive_build(dim_index + 1, coords + [i / dimension_resolution[dim_index]])
                for i in range(dimension_resolution[dim_index])]

    # Start recursive noise generation from dimension 0 and an empty coordinate list
    return recursive_build(0, [])






def perlin_matrix_to_coords(perlin_matrix, noise_cutoff = None):
    # Initialize empty lists for coordinates and values
    coordinates = []
    values = []

    # Helper function to recursively flatten the nested data
    def recursive_flatten(current_data, current_coords):
        for i, item in enumerate(current_data):
            if isinstance(item, list):
                # If the item is a list, recursively flatten it
                recursive_flatten(item, current_coords + [i])
            else:
                # If the item is a value, store its coordinates and value
                coordinates.append(current_coords + [i])
                values.append(item)

    # Start the recursive flattening process
    recursive_flatten(perlin_matrix, [])

    # Determine the number of dimensions
    num_dimensions = max(map(len, coordinates))

    # Create separate arrays for each dimension
    coordinate_arrays = [np.array([coord[dim] if dim < len(coord) else 0 for coord in coordinates]) for dim in range(num_dimensions)]

    # Convert the values list to a NumPy array
    values_array = np.array(values)



    return coordinate_arrays, values_array






def plot_perlin_coord_values(coordinate_arrays, values_array, edgecolors=None):
    if len(coordinate_arrays) == 2:
        plt.scatter(coordinate_arrays[0], coordinate_arrays[1], c=values_array, cmap='Greys', edgecolors=edgecolors)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Values')
        plt.title('2D Scatter Plot')
        plt.show()
    if len(coordinate_arrays) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coordinate_arrays[0], coordinate_arrays[1], coordinate_arrays[2], c=values_array, cmap='Greys', edgecolors=edgecolors)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(ax.scatter(coordinate_arrays[0], coordinate_arrays[1], coordinate_arrays[2], c=values_array, cmap='Greys'), label='Values')
        plt.title('3D Scatter Plot')
        plt.show()
