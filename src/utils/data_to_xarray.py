import numpy as np
import xarray as xr
import pandas as pd

import ND_north_star.src.noise_generators.perlin_noise_generator as PNG
from ND_north_star.src.utils.sampling_function import random_sampler, dual_sampler
from ND_north_star.src.edge_detection.contour_points_2D import generate_boundary_splines







def create_perlin_dataset(num_images: int, dimensions: list, octave: int, random_num_samples: int, boundary_points: int):
    datasets = []
    passed_images = 0
    # attempts = 0

    while passed_images < num_images:
        # print(f'Attempt {attempts}')
        save_dict = perlin_to_dict(dimensions=dimensions, octave=octave, num_samples=random_num_samples, boundary_points=boundary_points)
        

        if save_dict is None:
            # print(f'Skipped attempt {attempts} due to failure')
            continue
        else:
            ds = save_dict_to_xarray(save_dict)

            datasets.append(ds)
            passed_images += 1
            # print(f'Successfully added image {passed_images}')
       

    # Add octaves as a data variable to each dataset
    for ds in datasets:
        octave_value = ds.attrs.get('octaves', None)
        if octave_value is not None:
            ds = ds.assign(octaves=octave_value)

    # Concatenate the datasets along the 'image' dimension
    combined_ds = xr.concat(datasets, dim=pd.Index(range(passed_images), name='image'))

    # # Now you can access the octaves from the combined dataset
    # print(combined_ds['octaves'].values, 'combined dataset octaves')

    # combined_ds = xr.concat(datasets, dim=pd.Index(range(passed_images), name='image'))




    return combined_ds




def perlin_to_dict(dimensions:list, octave:int, num_samples:int, boundary_points:int):

    dataset = PNG.normalized_perlin_data(dimension_resolution=dimensions, octaves=octave)
    dataset = generate_boundary_splines(dataset, shared_boundary_points=boundary_points)

    sampled_dataset = random_sampler(dataset, num_samples=num_samples)

    keep_keys = ['features', 'values', 'octaves', 'resolution']
    save_dict = {key: dataset[key] for key in keep_keys}

    sampled_keys = ['features', 'values']
    save_dict.update({key + '_sampled': sampled_dataset[key] for key in sampled_keys})

    boundary_points = []
    for _, bound_points in dataset['boundary_splines'].items():
        boundary_points.extend(bound_points)
    boundary_points = np.array(boundary_points)


    spline_x = boundary_points[:, 0]
    spline_y = boundary_points[:, 1]
    tolerance = 0.01
    upper_bound = 1 + tolerance
    lower_bound = 0 - tolerance
    if (max(spline_x) > upper_bound) or (max(spline_y) > upper_bound) or (min(spline_x) < lower_bound) or (min(spline_y) < lower_bound):
        # print(max(spline_x), max(spline_y), min(spline_x), min(spline_y))
        # print('Boundary points are not normalized')
        return None


    save_dict['boundary_splines'] = boundary_points

    # convert_to_numpy = ['features', 'values']
    for key in save_dict.keys():
        if key.startswith('features') or key.startswith('values'):
            save_dict[key] = np.array(save_dict[key])

    return save_dict









############################################################################################################
####################### Convert between xarray and dict ################################################################
############################################################################################################


def save_dict_to_xarray(save_dict:dict):

    features = save_dict['features']
    X0 = features[:, 0]
    X1 = features[:, 1]

    sampled_features = save_dict['features_sampled']
    X0_sampled = sampled_features[:, 0]
    X1_sampled = sampled_features[:, 1]

    values = save_dict['values']
    sampled_values = save_dict['values_sampled']

    resolution = save_dict['resolution']
    X0_resolution = resolution[0]
    X1_resolution = resolution[1]

    octaves = save_dict['octaves']

    boundary_splines = save_dict['boundary_splines']
    X0_boundary = boundary_splines[:, 0]
    X1_boundary = boundary_splines[:, 1]


    # Create xarray dataset
    data_vars = {
        'values': (('index',), values),
        'sampled_values': (('sampled_ind',), sampled_values),
    }

    coords = {
        'feature_ind': np.arange(len(values)),
        'X0': (('feature_ind',), X0),
        'X1': (('feature_ind',), X1),
        'boundary_index': np.arange(len(X0_boundary)),
        'X0_boundary': (('boundary_index',), X0_boundary),
        'X1_boundary': (('boundary_index',), X1_boundary),
        'sampled_ind': np.arange(len(X0_sampled)),
        'X0_sampled': (('sampled_ind',), X0_sampled),
        'X1_sampled': (('sampled_ind',), X1_sampled),
    }

    attrs = {
        'X0_resolution': X0_resolution,
        'X1_resolution': X1_resolution,
        'octaves': octaves
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return ds





def xarray_to_dict(ds):
    # Extract data variables
    values = ds['values'].values

    # Extract features
    X0 = ds['X0'].values
    X1 = ds['X1'].values
    features = np.stack([X0, X1], axis=1)

    # Extract sampled features
    X0_sampled = ds['X0_sampled'].values
    X1_sampled = ds['X1_sampled'].values
    features_sampled = np.stack([X0_sampled, X1_sampled], axis=1)
    values_sampled = ds['sampled_values'].values

    # Extract boundary splines
    X0_boundary = ds['X0_boundary'].values
    X1_boundary = ds['X1_boundary'].values
    boundary_splines = np.stack([X0_boundary, X1_boundary], axis=1)

    # Extract resolution from attributes
    X0_resolution = ds.attrs['X0_resolution']
    X1_resolution = ds.attrs['X1_resolution']
    resolution = [X0_resolution, X1_resolution]

    # Extract octaves from attributes
    octaves = ds.attrs['octaves']

    # Create the dictionary to match the original structure
    save_dict = {
        'features': features,
        'features_sampled': features_sampled,
        'values': values,
        'values_sampled': values_sampled,
        'resolution': resolution,
        'octaves': octaves,
        'boundary_splines': boundary_splines,
        'dimension': len(resolution)
    }

    return save_dict