import numpy as np
import pandas as pd

import ND_north_star.src.noise_generators.perlin_noise_generator as PNG
from ND_north_star.src.utils.sampling_function import random_sampler, dual_sampler
from ND_north_star.src.edge_detection.contour_points_2D import generate_boundary_splines







def create_perlin_dict_dataset(num_images: int, dimensions: list, octave: int, random_num_samples: int, boundary_points: int):
    dict_datasets = []
    passed_images = 0
    # attempts = 0

    while passed_images < num_images:
        # print(f'Attempt {attempts}')
        save_dict = perlin_to_dict(dimensions=dimensions, octave=octave, num_samples=random_num_samples, boundary_points=boundary_points)
        
        if save_dict is None:
            # print(f'Skipped attempt {attempts} due to failure')
            continue
        else:
            
            dict_datasets.append(save_dict)
            passed_images += 1
            # print(f'Successfully added image {passed_images}')
       
    return dict_datasets




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






