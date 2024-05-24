# import numpy as np
# from sklearn.neighbors import KNeighborsRegressor

# from ND_north_star.src.data_fillers.find_empty_points import find_empty_data_points


# def kNN_data_filler(sparse_dataset_dict:dict, k:int = 3, rounding_threshold:float=0.5):

#     sparse_features = sparse_dataset_dict['features']
#     sparse_values = sparse_dataset_dict['values']
#     sparse_resolution = sparse_dataset_dict['resolution']

#     knn = KNeighborsRegressor(n_neighbors=k)
#     knn.fit(sparse_features, sparse_values)


#     # Generate filled coordinate lists for each resolution
#     array_per_dim = [np.linspace(0, 1, res) for res in sparse_resolution]

#     # Create a meshgrid for the coordinates in all dimensions
#     # Create a meshgrid for the coordinates in all dimensions
#     meshgrid = np.meshgrid(*array_per_dim, indexing='ij')
#     # Flatten the meshgrid arrays and combine them into a feature matrix
#     full_feature_points = np.vstack([grid.flatten() for grid in meshgrid]).T

#     # Find missing points by checking which full_feature_points are not in sparse_features
#     sparse_set = set(map(tuple, sparse_features))
#     missing_points = np.array([point for point in full_feature_points if tuple(point) not in sparse_set])

#     # Predict values for the missing points using kNN
#     if missing_points.size > 0:
#         missing_values = knn.predict(missing_points)
#     else:
#         missing_values = np.array([])

#     # Combine the sparse dataset with the newly computed missing points and values
#     combined_features = np.vstack((sparse_features, missing_points))
#     combined_values = np.concatenate((sparse_values, missing_values))



#     # Round the points to a consistent precision (e.g., 8 decimal places)
#     precision = 8
#     rounded_full_feature_points = np.round(full_feature_points, precision)
#     rounded_combined_features = np.round(combined_features, precision)

#     # Create a dictionary to map each point to its index in full_feature_points
#     point_to_index = {tuple(point): idx for idx, point in enumerate(rounded_full_feature_points)}

#     # Sort the combined features based on the indices in full_feature_points
#     sorted_indices = np.array([point_to_index[tuple(point)] for point in rounded_combined_features])
#     sorted_order = np.argsort(sorted_indices)

#     # Apply the sorted order to combined features and values
#     sorted_combined_features = combined_features[sorted_order]
#     sorted_combined_values = combined_values[sorted_order]




#     # Apply the threshold to round values to either 0 or 1
#     rounded_combined_values = (sorted_combined_values >= rounding_threshold).astype(int)

#     combined_dataset_dict = sparse_dataset_dict.copy()
#     combined_dataset_dict['features'] = sorted_combined_features
#     combined_dataset_dict['values'] = rounded_combined_values

#     return combined_dataset_dict
    
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def kNN_data_filler(sparse_dataset_dict: dict, k: int = 3, rounding_threshold: float = 0.5):
    sparse_features = sparse_dataset_dict['features']
    sparse_values = sparse_dataset_dict['values']
    sparse_resolution = sparse_dataset_dict['resolution']

    # Fit kNN model
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(sparse_features, sparse_values)

    # Generate full coordinate grid
    full_grid = np.meshgrid(*[np.linspace(0, 1, res) for res in sparse_resolution], indexing='ij')
    full_feature_points = np.vstack([grid.flatten() for grid in full_grid]).T

    # Identify missing points and placeholders
    sparse_set = set(map(tuple, sparse_features))
    missing_points = []
    combined_values = np.empty(len(full_feature_points))

    for idx, point in enumerate(full_feature_points):
        if tuple(point) in sparse_set:
            value_idx = np.where((sparse_features == point).all(axis=1))[0][0]
            combined_values[idx] = sparse_values[value_idx]
        else:
            missing_points.append(point)

    # Predict missing values
    if missing_points:
        missing_points = np.array(missing_points)
        missing_values = knn.predict(missing_points)
    else:
        missing_values = np.array([])

    # Fill in the combined values array
    missing_idx = 0
    for idx, point in enumerate(full_feature_points):
        if tuple(point) not in sparse_set:
            combined_values[idx] = missing_values[missing_idx]
            missing_idx += 1

    # Apply rounding threshold
    rounded_combined_values = (combined_values >= rounding_threshold).astype(int)

    # Create and return combined dataset
    combined_dataset_dict = sparse_dataset_dict.copy()
    combined_dataset_dict['features'] = full_feature_points
    combined_dataset_dict['values'] = rounded_combined_values

    return combined_dataset_dict
