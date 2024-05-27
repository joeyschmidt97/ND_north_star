import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev






def generate_boundary_splines(dataset:dict, smoothing_factor=0.001, num_points=100):
    boundary_points = find_boundary_points(dataset)
    distinct_boundaries = group_distinct_boundary_curves(boundary_points)

    dataset['distinct_boundary_points'] = distinct_boundaries

    boundary_splines = {}
    for boundary_ind, boundary_points in distinct_boundaries.items():
        if len(boundary_points) < 4:
            print(f"Skipping boundary {boundary_ind}: Not enough points to fit a spline.")
            continue
        x_new, y_new = generate_spline_curve(boundary_points, smoothing_factor, num_points)
        boundary_splines[boundary_ind] = np.array([x_new, y_new]).T

    return boundary_splines







def find_boundary_points(dataset:dict):
    points = dataset['features']
    values = dataset['values']

    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]


    # Interpolate data onto grid
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # Suppress the plot display
    plt.ioff()
    fig, ax = plt.subplots()

    # Find contour at halfway value (0.5) without displaying the plot
    contour = ax.contour(grid_x, grid_y, grid_z, levels=[0.5])

    # Extract the contour points
    contour_paths = contour.get_paths()
    contour_points = contour_paths[0].vertices

    # Close the figure to clean up
    plt.close(fig)
    
    return np.array(contour_points)







def group_distinct_boundary_curves(boundary_points):

    # Use DBSCAN to find clusters of points
    dbscan = DBSCAN(eps=0.03, min_samples=3)  # Adjust eps as needed

    points = boundary_points
    clusters = dbscan.fit_predict(boundary_points)



    # Create a dictionary to hold distinct boundaries
    distinct_boundaries = {}
    for idx, label in enumerate(clusters):
        if label not in distinct_boundaries:
            distinct_boundaries[label] = []
        distinct_boundaries[label].append(points[idx])

    # Convert the lists to numpy arrays for easier manipulation later
    distinct_boundaries = {k: np.array(v) for k, v in distinct_boundaries.items()}

    return distinct_boundaries







def generate_spline_curve(points, smoothing_factor=0.001, num_points=100, noise_level=1e-6):

    # Split data into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Check for unique points and add noise if necessary
    if len(np.unique(x)) < len(x):
        # print("Duplicate x values found. Adding noise to make them unique.")
        x += np.random.uniform(-noise_level, noise_level, size=x.shape)
    if len(np.unique(y)) < len(y):
        # print("Duplicate y values found. Adding noise to make them unique.")
        y += np.random.uniform(-noise_level, noise_level, size=y.shape)

    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isinf(x)) or np.any(np.isinf(y)):
        print("Invalid input: NaN or Inf values found. Skipping.")
    

    # Parameterize the points with a parameter t
    t = np.linspace(0, 1, len(points))

    # Fit splines to x(t) and y(t) with smoothing factor s
    # smoothing_factor = .001  # Adjust this value for more or less smoothing
    tck, u = splprep([x, y], s=smoothing_factor)

    # Evaluate the spline fit at a dense set of parameter values
    # num_points = 100  # Number of points in the final smoothed curve
    x_new, y_new = splev(np.linspace(0, 1, num_points), tck)

    return x_new, y_new