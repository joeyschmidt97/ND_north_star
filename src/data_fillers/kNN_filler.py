
import numpy as np
import copy


def kNN_data_filler(dataset_dict:dict, k:int = 3, resolution_list:list = None):
    """

    """
    # Check if the images have the same shape
    real_coord_list = dataset_dict['coordinates_list']
    real_values = dataset_dict['values_array']



    # Generate filled coordinate lists for each resolution
    ind_coord_arrays = [np.linspace(0, 1, res) for res in resolution_list]

    # Create a meshgrid for the coordinates in all dimensions
    meshgrid = np.meshgrid(*ind_coord_arrays, indexing='ij')

    # Flatten the meshgrid arrays and store each as a separate array in a list
    full_coord_list = [grid.flatten() for grid in meshgrid]

    # computed_coord_list = full_coord_list.copy()
    remove_indices = []

    # Iterate through the points in real_coord_list and full_coord_list
    for real_ind, _ in enumerate(real_coord_list[0]):
        real_point = [real_coord[real_ind] for real_coord in real_coord_list]
        real_point = tuple(real_point)

        for full_ind, _ in enumerate(full_coord_list[0]):
            full_point = [full_coord[full_ind] for full_coord in full_coord_list]
            full_point = tuple(full_point)

            

            if real_point == full_point:
                # print(real_point, full_point, real_point == full_point)

                remove_indices.append(full_ind)

                # for computed_dim, _ in enumerate(computed_coord_list):
                #     remove_indices
                #     computed_coord_list[computed_dim][full_ind].pop()
    
    computed_coord_list = full_coord_list.copy()
    for computed_dim, _ in enumerate(computed_coord_list):
        computed_coord_list[computed_dim] = np.delete(computed_coord_list[computed_dim], remove_indices)

    return computed_coord_list


