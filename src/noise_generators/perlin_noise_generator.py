import numpy as np
import matplotlib.pyplot as plt

from perlin_noise import PerlinNoise
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


from ND_north_star.src.utils.coord_to_image_transforms import coord_val_to_image


def normalized_perlin_coord_values(dimension_resolution:list, octaves:int):
    """
    Generate N-dimensional Perlin noise normalized between (0,1) of any octave (noisiness)

    Args:
    - dimension_resolution (list of int): The resolution along each dimension ([40,40] will generate 2D grid of spatial resolution 40 x 40 pixels).
    - octaves (int): The number of octaves to generate the noise (higher -> more jagged).

    Returns:
    - list of arrays: A list of arrays containing the normalized coordinates for each dimension.
    - array: An array containing the normalized noise values.
    """

    perlin_noise = ND_perlin_matrix(dimension_resolution=dimension_resolution, octaves=octaves, noise_cutoff_list=[0.5,0,1])
    coord_array, values = perlin_matrix_to_coords(perlin_noise)

    normalized_coord_array = normalize_coords(coord_array)

    return normalized_coord_array, values




############################################################################################################
######################################### Perlin noise generator ###########################################
############################################################################################################



def ND_perlin_matrix(dimension_resolution:list, octaves:int, noise_rescaling:list=[0, 1], noise_cutoff_list:list=None):
    """
    Generate N-dimensional Perlin noise of any octave (noisiness) with the ability to scale the nosie values (default 0-1) and noise cutoffs (round up/down to desired value)

    Args:
    - dimension_resolution (list of int): The resolution along each dimension ([40,40] will generate 2D grid of spatial resolution 40 x 40 pixels).
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






def perlin_matrix_to_coords(perlin_matrix):
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




def normalize_coords(input_coord_array):

    normalized_coord_array = []

    for coord in input_coord_array:
        min_val = coord.min()
        max_val = coord.max()
        range_val = max_val - min_val
        # Avoid division by zero in case all elements are the same
        if range_val != 0:
            normalized = (coord - min_val) / range_val
        else:
            normalized = np.zeros_like(coord)
        normalized_coord_array.append(normalized)

    return normalized_coord_array







def perlin_M_to_array_of_arrays(pic_array):
    pic_array = np.array(pic_array)

    nrows, ncols = pic_array.shape
    transformed_coordinates = []

    # Define the scaling function
    scale = lambda x, max_val: (x / max_val) * 100

    for y in range(nrows):
        for x in range(ncols):
            if pic_array[y, x] == 1:
                # Apply the scaling to each coordinate
                transformed_x = scale(x, ncols - 1)
                transformed_y = scale(y, nrows - 1)
                transformed_coordinates.append([transformed_x, transformed_y])

    return np.array(transformed_coordinates)
    







############################################################################################################
################################# Plotting functions for 2D and 3D #########################################
############################################################################################################



def plot_perlin_2D_3D(coordinate_arrays, values_array, edgecolors=None):

    if len(coordinate_arrays) == 2:
        fig = plt.figure()

        x_min = coordinate_arrays[0].min()
        x_max = coordinate_arrays[0].max()
        y_min = coordinate_arrays[1].min()
        y_max = coordinate_arrays[1].max()

        z_grid = coord_val_to_image(coordinate_arrays, values_array)

        print(z_grid)

        # Create a custom colormap
        cmap = mcolors.ListedColormap(['gray', 'black'])
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(6, 6))  # Set a consistent figure size
        plt.imshow(z_grid, cmap=cmap, norm=norm, origin='lower', extent=(x_min, x_max, y_min, y_max))
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



# def plot_perlin_2D_3D(coordinate_arrays, values_array, edgecolors=None):


#     if len(coordinate_arrays) == 2:
#         plt.scatter(coordinate_arrays[0], coordinate_arrays[1], c=values_array, cmap='Greys', edgecolors=edgecolors)
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.colorbar(label='Values')
#         plt.title('2D Scatter Plot')
#         plt.show()
#     if len(coordinate_arrays) == 3:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(coordinate_arrays[0], coordinate_arrays[1], coordinate_arrays[2], c=values_array, cmap='Greys', edgecolors=edgecolors)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         plt.colorbar(ax.scatter(coordinate_arrays[0], coordinate_arrays[1], coordinate_arrays[2], c=values_array, cmap='Greys'), label='Values')
#         plt.title('3D Scatter Plot')
#         plt.show()
