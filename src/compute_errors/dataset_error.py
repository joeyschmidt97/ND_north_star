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

        coord_compare = np.array(orig_coord) == np.array(pred_coords)
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
    half_hypercube_diag = np.sqrt(dimension)

    shape = shape.lower()
    if shape == 'circle':
        radii = np.linspace(0.01 * half_hypercube_diag, half_hypercube_diag, 50)
        sphere_center_coord = np.full(dimension, 0.5)

        MSE_inside = np.zeros_like(radii)
        MSE_outside = np.zeros_like(radii)

        for r_ind, r in enumerate(radii):
            SE_inside = []
            SE_outside = []
            
            for i in range(len(SE_values)):
                point = np.array([coord[i] for coord in coord_list])
                distance = np.linalg.norm(point - sphere_center_coord)

                if distance <= r:
                    SE_inside.append(SE_values[i])
                else:
                    SE_outside.append(SE_values[i])
            

            if SE_inside:
                MSE_inside[r_ind] = np.mean(SE_inside)
            else:
                MSE_inside[r_ind] = np.nan  # or another value to indicate no data

            if SE_outside:
                MSE_outside[r_ind] = np.mean(SE_outside)
            else:
                MSE_outside[r_ind] = np.nan  # or another value to indicate no data


        plt.plot(radii/half_hypercube_diag, MSE_inside, label='Inside Sphere')
        plt.plot(radii/half_hypercube_diag, MSE_outside, label='Outside Sphere')
        plt.xlabel('$R/l_{diag}$')
        plt.ylabel('MSE')
        plt.legend()
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
        plt.title('2D Image Plot')

        # Set x and y limits from 0 to 1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Ensure equal scaling for both axes
        ax.set_aspect('equal')
        plt.show()
    

