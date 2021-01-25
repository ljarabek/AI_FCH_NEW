import numpy as np
from scipy.interpolate import RectBivariateSpline

def interpolate2d(image, new_size: tuple):
    coordinates_1 = np.arange(0, image.shape[0], 1)
    coordinates_2 = np.arange(0, image.shape[1], 1)
    indices = (np.linspace(0, image.shape[0], new_size[0]), np.linspace(0, image.shape[1], new_size[1]))
    # print(indices[0].shape)

    spline = RectBivariateSpline(coordinates_1, coordinates_2, image)

    image = np.array(spline(indices[0], indices[1], grid=True))

    return image

def interpolate3d(arr: np.ndarray) -> np.ndarray:
    raise NotImplementedError