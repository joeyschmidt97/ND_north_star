import numpy as np
from scipy.spatial import cKDTree

from ND_north_star.src.edge_detection.contour_points_2D import find_boundary_points


def random_sampler(dataset_dict:dict, num_samples:int):
    """

    """
    features = np.array(dataset_dict['features'])
    values = np.array(dataset_dict['values'])
    
    random_indices = np.random.choice(len(features), size=num_samples, replace=False)

    # Sampled features and values
    sampled_features = features[random_indices]
    sampled_values = values[random_indices]

    sampled_dataset_dict = dataset_dict.copy()
    sampled_dataset_dict['features'] = sampled_features
    sampled_dataset_dict['values'] = sampled_values

    return sampled_dataset_dict



def dual_sampler(dataset_dict:dict, num_samples, random_sample_perc:float=0.5):
    """

    """
    features = np.array(dataset_dict['features'])
    values = np.array(dataset_dict['values'])
    
    num_random_samples = int(num_samples * random_sample_perc)
    random_indices = np.random.choice(len(features), size=num_random_samples, replace=False)
    random_sampled_features = features[random_indices]
    random_sampled_values = values[random_indices]

    rand_sampled_dataset_dict = dataset_dict.copy()
    rand_sampled_dataset_dict['features'] = random_sampled_features
    rand_sampled_dataset_dict['values'] = random_sampled_values

    print(rand_sampled_dataset_dict)



    # Contour edge sampling
    num_contour_samples = num_samples - num_random_samples
    contour_points = find_boundary_points(rand_sampled_dataset_dict) # CANT SAMPLED FROM FULL DATA MUST BE FROM RANDOM SAMPLED DATA

    # Introduction while loop to sample more random points if num_contour_samples > len(contour_points)
    # print out new percentage of random samples and contour samples if this happens

    if num_contour_samples > len(contour_points):
        num_contour_samples = len(contour_points)

    contour_indices = np.random.choice(len(contour_points), size=num_contour_samples, replace=False)
    contour_points_sampled = contour_points[contour_indices]


    # Build a KD-Tree for fast nearest-neighbor lookup
    tree = cKDTree(features)

    # Find the nearest feature point for each contour point
    _, nearest_feature_indices = tree.query(contour_points_sampled)

    # Get the corresponding feature points and values
    contour_sampled_features = features[nearest_feature_indices]
    contour_sampled_values = values[nearest_feature_indices]



    # Sampled features and values
    sampled_features = np.concatenate([random_sampled_features, contour_sampled_features])
    sampled_values = np.concatenate([random_sampled_values, contour_sampled_values])


    sampled_dataset_dict = dataset_dict.copy()
    sampled_dataset_dict['features'] = sampled_features
    sampled_dataset_dict['values'] = sampled_values

    return sampled_dataset_dict