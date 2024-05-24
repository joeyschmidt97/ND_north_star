import numpy as np

def gradient_edge_detection(data):
    """Detect edges using gradient."""
    gx = np.diff(data, axis=0, prepend=0)
    gy = np.diff(data, axis=1, prepend=0)
    edges = np.logical_or(gx != 0, gy != 0)
    
    return edges


