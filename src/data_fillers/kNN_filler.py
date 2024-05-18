import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from ND_north_star.src.data_fillers.find_empty_points import find_empty_data_points


def kNN_data_filler(sparse_dataset_dict:dict, k:int = 3, resolution_list:list = 'auto'):

    sparse_features = sparse_dataset_dict['features']
    sparse_values = sparse_dataset_dict['values']
    sparse_resolution = sparse_dataset_dict['resolution']

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(sparse_features, sparse_values)



    # Generate filled coordinate lists for each resolution
    array_per_dim = [np.linspace(0, 1, res) for res in sparse_resolution]

    # Create a meshgrid for the coordinates in all dimensions
    # Create a meshgrid for the coordinates in all dimensions
    meshgrid = np.meshgrid(*array_per_dim, indexing='ij')
    # Flatten the meshgrid arrays and combine them into a feature matrix
    full_feature_points = np.vstack([grid.flatten() for grid in meshgrid]).T

    # Find missing points by checking which full_feature_points are not in sparse_features
    sparse_set = set(map(tuple, sparse_features))
    missing_points = np.array([point for point in full_feature_points if tuple(point) not in sparse_set])

    # Predict values for the missing points using kNN
    if missing_points.size > 0:
        missing_values = knn.predict(missing_points)
    else:
        missing_values = np.array([])

    # Combine the sparse dataset with the newly computed missing points and values
    combined_features = np.vstack((sparse_features, missing_points))
    combined_values = np.concatenate((sparse_values, missing_values))


    combined_dataset_dict = sparse_dataset_dict.copy()
    combined_dataset_dict['features'] = combined_features
    combined_dataset_dict['values'] = combined_values

    return combined_dataset_dict
    