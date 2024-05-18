import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors




def compute_ND_error(original_ND_data:dict, predicted_ND_data:dict):
    """
    Compute the squared error between two dataset
    """
    # Check if the images have the same shape
    original_coord_list = np.array(original_ND_data['features'])
    predicted_coord_list = np.array(predicted_ND_data['features'])

    assert len(original_coord_list) == len(predicted_coord_list), "Compared datasets must the same number of points"

    coord_compare = np.isclose(original_coord_list, predicted_coord_list)
    assert coord_compare.all(), "Compared datasets must have the same coordinates to compare"

    original_values = np.array(original_ND_data['values'])
    predicted_values = np.array(predicted_ND_data['values'])
    
    SE_data = (original_values - predicted_values)**2

    SE_dataset_dict = original_ND_data.copy()
    SE_dataset_dict['values'] = SE_data

    return SE_dataset_dict





def plot_SE_ND_in_out_plots(SE_dataset_dict:dict, shape:str = 'circle'):

    point_list = SE_dataset_dict['features']
    SE_values = SE_dataset_dict['values']

    dimension = SE_dataset_dict['dimension']
    half_hypercube_diag = np.sqrt(dimension)/2

    shape = shape.lower()
    if shape == 'circle':
        radii = np.linspace(0.01 * half_hypercube_diag, half_hypercube_diag, 100)
        sphere_center_coord = np.full(dimension, 0.5)

        distances = np.linalg.norm(point_list - sphere_center_coord, axis=1)

        MSE_inside = np.zeros_like(radii)
        MSE_outside = np.zeros_like(radii)

        for r_ind, r in enumerate(radii):
            inside_mask = distances <= r
            outside_mask = ~inside_mask

            SE_inside = SE_values[inside_mask]
            SE_outside = SE_values[outside_mask]
            
            MSE_inside[r_ind] = np.mean(SE_inside) if SE_inside.size > 0 else 0
            MSE_outside[r_ind] = np.mean(SE_outside) if SE_outside.size > 0 else 0

        norm_radii = radii / half_hypercube_diag
        max_circle_radii = 0.5 / half_hypercube_diag

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

        ax1.plot(norm_radii, MSE_inside, label=f'Inside {dimension}D-hypersphere')
        ax1.axvline(x=max_circle_radii, color='r', linestyle='--', label=f'Max {dimension}D-hypersphere Radius')
        ax1.set_xlabel('$R/l_{diag}$')
        ax1.set_ylabel('MSE')
        ax1.legend()

        dMSE_inside = np.gradient(MSE_inside, norm_radii)
        dMSE_outside = np.gradient(MSE_outside, norm_radii)

        ax2.plot(norm_radii, dMSE_inside, label=f'Inside {dimension}D-hypersphere')
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