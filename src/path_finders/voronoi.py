import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d



def voronoi_points(coord_array, vals, threshold, plot=False):

    filtered_coord = []
    filtered_vals = []
    threshold = 0.5

    for ind,v in enumerate(vals):
        if v > threshold:
            filtered_vals.append(v)

            temp_coord = [coord[ind] for coord in coord_array]
            filtered_coord.append(temp_coord)

    # Compute Voronoi diagram
    vor = Voronoi(filtered_coord)
    
    if plot:
        # Plotting
        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)
        # ax.plot(filtered_coord[:, 0], filtered_coord[:, 1], 'ko')
        
        # Configure the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

