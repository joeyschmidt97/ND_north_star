import numpy as np


def coord_val_to_image(coord_array:list, values_array:list):
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





