import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from perlin_noise import PerlinNoise



def normalized_perlin_data(dimension_resolution: list, octaves: int):
    # Assuming ND_perlin_matrix is a function that generates the perlin noise matrix
    perlin_matrix = ND_perlin_matrix(dimension_resolution=dimension_resolution, octaves=octaves, noise_cutoff_list=[0.5, 0, 1])
    
    # Convert list to numpy array if needed
    array = np.array(perlin_matrix)
    shape = array.shape
 
    # Generate all possible coordinates in the N-dimensional array
    coordinates = np.indices(shape).reshape(len(shape), -1).T
    data = []      
    # Iterate over the coordinates and get the corresponding values
    for coord in coordinates:
        value = array[tuple(coord)]
        data.append(list(coord) + [value])
    
    # Separate the features and the value column
    features = [d[:-1] for d in data]
    values = [d[-1] for d in data]
    
    # Normalize the feature columns
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Create the output dictionary
    perlin_dict = {
        'features': normalized_features.tolist(),
        'values': values,
        'resolution': dimension_resolution,
        'coordinates': [f'x{i}' for i in range(len(shape))],
        'octaves': octaves,
        'dimension': len(dimension_resolution),
    }
    
    return perlin_dict




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






############################################################################################################
################################# Plotting functions for 2D and 3D #########################################
############################################################################################################



def plot_perlin_2D_3D(dataset_dict:dict, edgecolors=None, cmap='gray_r'):

    features = np.array(dataset_dict['features'])
    values = dataset_dict['values']
    coordinates = dataset_dict['coordinates']
    resolution_list = dataset_dict['resolution']
    D = dataset_dict['dimension']

    print(features[0:5])

    if D == 2:
        plt.figure(figsize=(8, 8))
        resolution = min(resolution_list)

        if resolution < 100:
            marker_scale = resolution
        else:
            marker_scale = 4*resolution

        print(features[:,0][0:5])
        print(features[:,1][0:5])
        print(values[0:5])
        # Scatter plot
        scatter = plt.scatter(features[:, 0],features[:, 1], c=values, cmap=cmap, s=1200/marker_scale, edgecolors=edgecolors)

        # Adding a color bar to show the mapping from values to colors
        plt.colorbar(scatter, label='Value Intensity')

        # Labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        # plt.title('2D Scatter Plot with Grayscale Values')

        # Show plot
        plt.show()

    
    # if D == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(coordinate_arrays[0], coordinate_arrays[1], coordinate_arrays[2], c=values_array, cmap='Greys', alpha=0.2)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.colorbar(ax.scatter(coordinate_arrays[0], coordinate_arrays[1], coordinate_arrays[2], c=values_array, cmap='Greys'), label='Values')
    #     plt.title('3D Scatter Plot')
    #     plt.show()

