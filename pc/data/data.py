import enum

import pandas as pd
import pymia.data.conversion as conv
import numpy as np

KEY_POINT_FEATURES = 'point_features'
KEY_IMAGE_INFORMATION = 'image_information'
KEY_IMAGE_INFORMATION_LABELS = 'image_information_labels'


class FileTypes(enum.Enum):
    COORDINATE = 1  # The normalized (x, y, z) point cloud
    #POINT_FEATURES = 2  # The predicted probabilities associated with the point cloud
    IMAGE_INFORMATION = 3  # The image evidence (probabilities and Hessian entires)
    LABEL = 4  # The label associated with the point cloud
    GTM = 5  # The ground truth image  todo: rename
    INDICES = 6  # The (x, y, z) coordinates in the image domain
    #IMAGE_INFORMATION_LABEL = 7
    #LABEL_DISTANCE_MAP = 8


def point_cloud_to_image(cloud: np.ndarray, indices: np.ndarray, properties: conv.ImageProperties):
    """Converts a point cloud to a 3-D image.

    Args:
        cloud: The point cloud of shape (N, 3) with N being the number of points.
        indices: The indices
        properties: The properties of the 3-D image.
    """
    img_arr = np.zeros(properties.size[::-1]) if cloud.ndim == 1 else \
        np.zeros(properties.size[::-1] + (cloud.shape[-1], ))
    for idx in range(cloud.shape[0]):
        x, y, z = indices[idx]
        img_arr[z, y, x] = cloud[idx]
    return conv.NumpySimpleITKImageBridge.convert(img_arr, properties)


def save_to_csv(path, data, color, properties: conv.ImageProperties):
    """Saves a point cloud to a CSV file.

    Args:
        path: The CSV file path.
        data: The data of shape (N, 3) with N being the number of points.
        color: The color of each point of shape (N, 1). If `None`, then np.ones(N, 1) is used.
        properties: The image properties.
    """

    def transform_to_physical_coordinates(index):
        """Transforms an image index to physical coordinates.

        The transformation is given by x = D * S * v + o
        where x is coordinate of the voxel in physical space, v is voxel index, o is origin,
        D is direction matrix, and S is diag (spacing).

        Args:
            index: The voxel index.

        Returns:
            The physical coordinates.
        """
        return np.matmul(
            np.matmul(np.array(properties.direction).reshape(3, 3), np.diag(properties.spacing)),
            index) + properties.origin

    data = np.apply_along_axis(transform_to_physical_coordinates, 1, data)

    if color is not None:
        data = np.concatenate([data, color], -1)

    pd.DataFrame(data).to_csv(path,
                              header=['x', 'y', 'z'] if color is None else ['x', 'y', 'z', 'intensity'],
                              index=None)
