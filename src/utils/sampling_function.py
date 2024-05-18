import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

def random_sampler(dataset_dict:dict, num_samples, method='random'):
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
