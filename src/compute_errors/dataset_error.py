import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ND_north_star.src.utils.coord_to_image_transforms import dataset_2D_to_image




def compute_ND_error(original_ND_data:dict, predicted_ND_data:dict):
    """
    Compute the squared error between two dataset
    """
    # Check if the images have the same shape
    original_coord_list = original_ND_data['coordinates_list']
    predicted_coord_list = predicted_ND_data['coordinates_list']

    assert len(original_coord_list) == len(predicted_coord_list), "Compared datasets must have the same number of dimensions"

    for orig_coord, pred_coords in zip(original_coord_list, predicted_coord_list):
        assert len(orig_coord) == len(pred_coords), "Compared datasets must have the same number of coordinates"

        coord_compare = np.isclose(orig_coord, pred_coords)
        assert coord_compare.all(), "Compared datasets must have the same coordinates to compare"

    original_values = original_ND_data['values_array']
    predicted_values = predicted_ND_data['values_array']
    
    SE_data = (original_values - predicted_values)**2

    SE_dataset_dict = {'coordinates_list': original_coord_list, 'values_array': SE_data}

    return SE_dataset_dict





def plot_SE_ND_in_out_plots(SE_dataset_dict:dict, shape:str = 'circle'):

    coord_list = SE_dataset_dict['coordinates_list']
    SE_values = SE_dataset_dict['values_array']

    dimension = len(coord_list)
    half_hypercube_diag = np.sqrt(dimension)/2

    shape = shape.lower()
    if shape == 'circle':
        # radii = np.linspace(0.01, 0.5, 100)
        radii = np.linspace(0.01 * half_hypercube_diag, half_hypercube_diag, 100)
        sphere_center_coord = np.full(dimension, 0.5)

        MSE_inside = np.zeros_like(radii)
        MSE_outside = np.zeros_like(radii)

        for r_ind, r in enumerate(radii):
            SE_inside = np.zeros_like(SE_values)
            SE_outside = np.zeros_like(SE_values)
            
            for i in range(len(SE_values)):

                point = np.array([coord[i] for coord in coord_list])
                distance = np.linalg.norm(point - sphere_center_coord)

                # print(r, half_hypercube_diag, point, distance)

                if distance <= r:
                    SE_inside[i] = SE_values[i]
                else:
                    SE_outside[i] = SE_values[i]
            
            MSE_inside[r_ind] = np.mean(SE_inside)
            MSE_outside[r_ind] = np.mean(SE_outside)

        norm_radii = radii / half_hypercube_diag
        max_circle_radii = 0.5 / half_hypercube_diag
        # norm_radii = radii / 0.5

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

        ax1.plot(norm_radii, MSE_inside, label=f'Inside {dimension}D-hypersphere')
        # ax1.plot(norm_radii, MSE_outside, label=f'Outside {dimension}D-hypersphere')
        ax1.axvline(x=max_circle_radii, color='r', linestyle='--', label=f'Max {dimension}D-hypersphere Radius')
        ax1.set_xlabel('$R/l_{diag}$')
        ax1.set_ylabel('MSE')
        ax1.legend()

        dMSE_inside = np.gradient(MSE_inside, norm_radii)
        dMSE_outside = np.gradient(MSE_outside, norm_radii)

        ax2.plot(norm_radii, dMSE_inside, label=f'Inside {dimension}D-hypersphere')
        # ax2.plot(norm_radii, dMSE_outside, label=f'Outside {dimension}D-hypersphere')
        ax2.axvline(x=max_circle_radii, color='r', linestyle='--', label=f'Max {dimension}D-hypersphere Radius')
        ax2.set_xlabel('$R/l_{diag}$')
        ax2.set_ylabel('d(MSE)/d($R/l_{diag}$)')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    elif shape == 'square':
        radii = np.linspace(0, half_hypercube_diag, 100)
        pass
    else:
        raise ValueError("Invalid shape. Please choose 'circle' or 'square'")




def plot_SE_2D(SE_dataset_dict:dict):
    """
    Compute the Mean Squared Error between two images
    """
    coordinate_arrays = SE_dataset_dict['coordinates_list']
    
    if len(coordinate_arrays) == 2:
        z_grid = dataset_2D_to_image(SE_dataset_dict)
        fig = plt.figure()

        x_min = coordinate_arrays[0].min()
        x_max = coordinate_arrays[0].max()
        y_min = coordinate_arrays[1].min()
        y_max = coordinate_arrays[1].max()

        # Create a custom colormap
        cmap = mcolors.ListedColormap(['green', 'red'])
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(6, 6))  # Set a consistent figure size
        plt.imshow(z_grid, cmap=cmap, norm=norm, origin='lower', extent=(x_min, x_max, y_min, y_max))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Values', ticks=[0, 1])
        plt.title('2D Squared Error Image')

        # Set x and y limits from 0 to 1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Ensure equal scaling for both axes
        ax.set_aspect('equal')
        plt.show()
    

