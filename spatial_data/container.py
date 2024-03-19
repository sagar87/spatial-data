from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .constants import Dims, Layers


def load_image_data(
    image: np.ndarray,
    channel_coords: Union[str, List[str]],
    segmentation: Union[None, np.ndarray] = None,
    labels: Union[None, pd.DataFrame] = None,
    cell_col: str = "cell",
    label_col: str = "label",
    copy_segmentation: bool = False,
    copy_image: bool = False,
):
    """Creates a image container.

    Creates an Xarray dataset with images, segmentation, and
    coordinate fields.

    Parameters
    ----------
    image : np.ndarray
        np.ndarray with image.shape = (n, x, y)
    channel_coords: str | List[str]
        list with the names for each channel

    Returns
    -------
    xr.Dataset
        An X-array dataset with all fields.
    """
    if copy_image:
        image = image.copy()

    if type(channel_coords) is str:
        channel_coords = [channel_coords]

    if image.ndim == 2:
        image = np.expand_dims(image, 0)

    channel_dim, y_dim, x_dim = image.shape

    assert len(channel_coords) == channel_dim, "Length of channel_coords must match image.shape[0]."

    if labels is not None:
        assert segmentation is not None, "Labels may only be provided in conjunction with a segmentation."

    im = xr.DataArray(
        image,
        coords=[channel_coords, range(y_dim), range(x_dim)],
        dims=[Dims.CHANNELS, Dims.Y, Dims.X],
        name=Layers.IMAGE,
    )

    dataset = xr.Dataset(data_vars={Layers.IMAGE: im})

    if segmentation is not None:
        dataset = dataset.pp.add_segmentation(segmentation, copy=copy_segmentation).pp.add_observations()

        if labels is not None:
            dataset = dataset.pp.add_labels(labels, cell_col=cell_col, label_col=label_col)

    else:
        dataset = xr.Dataset(data_vars={Layers.IMAGE: im})

    return dataset
