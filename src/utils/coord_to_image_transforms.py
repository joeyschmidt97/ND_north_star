import numpy as np





def image_to_dataset_2D(image_2D: np.ndarray):
    """
    Transforms a 2D image representation into a set of coordinate and value arrays.

    Parameters:
    - image_2D: 2D numpy array
        A 2D array where each cell corresponds to a value, with NaNs indicating missing values.

    Returns:
    - coord_array: list
        A list containing two lists of normalized coordinates (x and y) between 0 and 1.
    - values_array: list
        A list containing the corresponding values for each (x, y) coordinate pair.
    """
    y_coords, x_coords = np.where(~np.isnan(image_2D))
    values = image_2D[y_coords, x_coords].astype(int)

    height, width = image_2D.shape

    x_array = x_coords / (width - 1)
    y_array = y_coords / (height - 1)

    coord_array = [x_array.tolist(), y_array.tolist()]
    values_array = values.tolist()

    dataset_dict = {'coordinates_array': coord_array, 'values_array': values} 

    return dataset_dict





def dataset_2D_to_image(dataset_dict:dict):
    """
    Transforms a set of coordinate and value arrays into a 2D image representation. Will default to NaNs if no values at a given coordinate are given.

    Parameters:
    - coord_array: list
        A list containing two lists of coordinates (x and y).
    - values_array: list
        A list containing the corresponding values for each (x, y) coordinate pair.

    Returns:
    - image_2D: 2D numpy array
        A 2D array where each cell corresponds to a value from the values_array, positioned according to their coordinates in the coord_array.

    Raises:
    - AssertionError: If the length of coord_array is not 2 or if the lengths of the x, y, and values arrays do not match.
    """
    copy_dataset_dict = dataset_dict.copy()
    coord_array = copy_dataset_dict['coordinates_list']
    values_array = copy_dataset_dict['values_array']

    
    assert len(coord_array) == 2, "Ensure there are only two coordinates in the 'coord_array' to transform to an image"

    x, y = coord_array
    z = values_array

    assert len(x) == len(y) == len(z), "The length of the coordinate arrays and values array must be the same"

    # Create a 2D grid of NaN values
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    image_2D = np.full((len(y_unique), len(x_unique)), np.nan)

    for i in range(len(x)):
        x_idx = np.where(x_unique == x[i])[0][0]
        y_idx = np.where(y_unique == y[i])[0][0]
        image_2D[y_idx, x_idx] = z[i]

    return image_2D





