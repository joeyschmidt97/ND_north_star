import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from ND_north_star.src.data_fillers.find_empty_points import find_empty_data_points


def kNN_data_filler(incomplete_dataset_dict:dict, k:int = 3, resolution_list:list = 'auto'):

    real_coord_list = incomplete_dataset_dict['coordinates_list']
    real_values = incomplete_dataset_dict['values_array']

    real_coord_array = np.array(real_coord_list).T # Convert from [x1,x2,x3,...] to [[x1_1,x2_1,x3_1,...], [x1_2,x2_2,x3_2,...], ...]


    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(real_coord_array, real_values)


    computed_dataset_dict = find_empty_data_points(incomplete_dataset_dict, resolution_list)
    full_coord_list = computed_dataset_dict['coordinates_list']
    full_values = computed_dataset_dict['values_array']

    # Create a mask for NaN values in full_values
    nan_mask = np.isnan(full_values)
    non_nan_mask = ~nan_mask # Invert the mask to get a mask for non-NaN values

    # Filter the coordinates based on the NaN mask
    compute_coords = np.array(full_coord_list).T[non_nan_mask]
    compute_values = knn.predict(compute_coords)



    combined_coords = np.array(full_coord_list).T
    combined_values = np.zeros_like(full_values)

    for i, coord in enumerate(combined_coords):
        coord_tuple = tuple(coord)

        if coord_tuple in map(tuple, real_coord_array):
            index = list(map(tuple, real_coord_array)).index(coord_tuple)
            combined_values[i] = real_values[index]
        elif coord_tuple in map(tuple, compute_coords):
            index = list(map(tuple, compute_coords)).index(coord_tuple)
            combined_values[i] = compute_values[index]

    # Create the combined dataset dictionary
    combined_dataset_dict = {
        'coordinates_list': full_coord_list,
        'values_array': combined_values
    }



    return combined_dataset_dict
    