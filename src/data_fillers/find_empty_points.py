
import numpy as np


def find_empty_data_points(incomplete_dataset_dict:dict, resolution_list:list = 'auto'):
    """

    """
    # Check if the images have the same shape
    real_coord_list = incomplete_dataset_dict['coordinates_list']


    if resolution_list == 'auto':
        # Calculate the differences between consecutive elements in each dimension
        differences_per_dim = [np.diff(coord) for coord in real_coord_list]

        # Find the minimum difference in each dimension
        min_coord_dist_per_dim = [np.min(diff[diff > 0]) for diff in differences_per_dim if np.any(diff > 0)]
        calc_res_per_dim = [int(1/min_dist) + 1 for min_dist in min_coord_dist_per_dim]
        
        print(f"Calculated resolution per dimension: {calc_res_per_dim}")

        res_per_dim = input("Enter the desired resolution per dimension as a list of integers (ex '60,60,60'): ")
        resolution_list = [int(res) for res in res_per_dim.split(",")]



    if isinstance(resolution_list, list) and len(resolution_list) == len(real_coord_list):
        res_per_dim = resolution_list
    else:
        raise ValueError("Resolution_list must be the same length as the number of dimensions in the dataset")



    # Generate filled coordinate lists for each resolution
    ind_coord_arrays = [np.linspace(0, 1, res) for res in resolution_list]

    # Create a meshgrid for the coordinates in all dimensions
    meshgrid = np.meshgrid(*ind_coord_arrays, indexing='ij')
    # Flatten the meshgrid arrays and store each as a separate array in a list
    full_coord_list = [grid.flatten() for grid in meshgrid]
    full_values_array = np.zeros_like(full_coord_list[0])

    # Iterate through the points in real_coord_list and full_coord_list
    for real_ind, _ in enumerate(real_coord_list[0]):
        real_point = [real_coord[real_ind] for real_coord in real_coord_list]
        real_point = tuple(real_point)

        for full_ind, _ in enumerate(full_coord_list[0]):
            full_point = [full_coord[full_ind] for full_coord in full_coord_list]
            full_point = tuple(full_point)

            if real_point == full_point:
                full_values_array[full_ind] = np.nan

    computed_dataset = {'coordinates_list': full_coord_list, 'values_array': full_values_array}

    return computed_dataset







def zero_filler(incomplete_dataset_dict:dict, resolution_list:list = 'auto'):

    computed_dataset_dict = find_empty_data_points(incomplete_dataset_dict, resolution_list)

    full_coord_list = computed_dataset_dict['coordinates_list']
    full_values = computed_dataset_dict['values_array']

    real_coord_list = incomplete_dataset_dict['coordinates_list']
    real_values = incomplete_dataset_dict['values_array']

    computed_values = full_values.copy()
    computed_values += 1

    for real_ind, _ in enumerate(real_coord_list[0]):
        real_point = [real_coord[real_ind] for real_coord in real_coord_list]
        real_point = tuple(real_point)

        for full_ind, _ in enumerate(full_coord_list[0]):
            full_point = [full_coord[full_ind] for full_coord in full_coord_list]
            full_point = tuple(full_point)

            if real_point == full_point:
                computed_values[full_ind] = real_values[real_ind]


    computed_dataset = {'coordinates_list': full_coord_list, 'values_array': computed_values}
    
    return computed_dataset

