

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def generate_binary_perlin_noise(dimensions, octaves):
    noise = PerlinNoise(octaves=octaves)

    def recursive_build(dim_index, coords):
        if dim_index == len(dimensions):
            return int(noise(coords) >= 0)
        
        return [recursive_build(dim_index + 1, coords + [i / dimensions[dim_index]])
                for i in range(dimensions[dim_index])]

    return recursive_build(0, [])




def plot_2d(pic):
    plt.imshow(pic, cmap='gray')
    plt.colorbar()
    plt.show()




def plot_3d(pic):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xpix, ypix, zpix = len(pic), len(pic[0]), len(pic[0][0])
    points = []

    # Extracting the points where value is 1
    for i in range(xpix):
        for j in range(ypix):
            for k in range(zpix):
                if pic[i][j][k] == 0:
                    points.append([i, j, k])

    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', marker='o')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()





