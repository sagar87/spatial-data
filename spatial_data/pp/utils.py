from typing import List, Union

import numpy as np
from skimage.segmentation import find_boundaries
from skimage import filters

from ..pl import _get_linear_colormap


def _render_label(mask, cmap_mask, img=None, alpha=0.2, alpha_boundary=1.0, mode="inner"):
    colored_mask = cmap_mask(mask)

    mask_bool = mask > 0
    mask_bound = np.bitwise_and(mask_bool, find_boundaries(mask, mode=mode))

    # blend
    if img is None:
        img = np.zeros(mask.shape + (4,), np.float32)
        img[..., -1] = 1

    im = img.copy()

    im[mask_bool] = alpha * colored_mask[mask_bool] + (1 - alpha) * img[mask_bool]
    im[mask_bound] = alpha_boundary * colored_mask[mask_bound] + (1 - alpha_boundary) * img[mask_bound]

    return im


def _label_segmentation_mask(segmentation: np.ndarray, annotations: dict) -> np.ndarray:
    """
    Relabels a segmentation according to the annotations df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    all_cells = []

    for k, v in annotations.items():
        mask = np.isin(segmentation, v)
        labeled_segmentation[mask] = k
        all_cells.extend(v)

    # remove cells that are not indexed
    neg_mask = ~np.isin(segmentation, all_cells)
    labeled_segmentation[neg_mask] = 0

    return labeled_segmentation


def _remove_segmentation_mask_labels(segmentation: np.ndarray, labels: Union[list, np.ndarray]) -> np.ndarray:
    """
    Relabels a segmentation according to the labels df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    mask = ~np.isin(segmentation, labels)
    labeled_segmentation[mask] = 0

    return labeled_segmentation


def _normalize(
    img: np.ndarray,
    pmin: float = 3.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
    name: str = "normed",
) -> np.ndarray:
    """Performs a min max normalisation.

    This function was adapted from the csbdeep package.

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    pmin: float
        Lower quantile (min value) used to perform qunatile normalization.
    pmax: float
        Upper quantile (max value) used to perform qunatile normalization.
    eps: float
        Epsilon float added to prevent 0 division.
    clip: bool
        Ensures that normed image array contains no values greater than 1.

    Returns
    -------
    xr.DataArray
        A min-max normalized image.
    """
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps)

    if clip:
        norm = np.clip(norm, 0, 1)

    return norm


def _colorize(
    img: np.ndarray,
    colors: List[str] = ["C1", "C2", "C3", "C4", "C5"],
    background: str = "black",
    normalize: bool = True,
    name: str = "colored",
) -> np.ndarray:
    """Colorizes a stack of images

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    clors: List[str]
        A list of strings that denote the color of each channel
    background: float
        Background color of the colorized image.
    normalize: bool
        Normalizes the image prior to colorizing it.

    Returns
    -------
    np.ndarray
        A colorized image
    """
    num_channels = img.shape[0]

    assert (
        len(colors) >= num_channels
    ), "Length of colors must at least be greater or equal the number of channels of the image."

    cmaps = _get_linear_colormap(colors[:num_channels], background)

    if normalize:
        img = _normalize(img)

    colored = np.stack([cmaps[i](img[i]) for i in range(num_channels)], 0)

    return colored


def merge(images, colors=["C1", "C2", "C3", "C4", "C5"], proj="sum", alpha=0.5):

    if proj == "sum":
        im_combined = np.sum(np.stack(images, axis=3), axis=3)
        im_combined[im_combined > 1] = 1
    elif proj == "blend":
        im_base = images[0].copy()
        for i in range(1, len(images)):
            alpha_a = images[i][:, :, 3][:, :, np.newaxis]
            alpha_a[alpha_a > 0] = alpha
            alpha_b = im_base[:, :, 3][:, :, np.newaxis]
            alpha_0 = alpha_a + alpha_b * (1 - alpha_a)
            im_combined = np.ones_like(images[0])
            im_combined[:, :, 0:3] = (
                images[i][:, :, 0:3] * alpha_a + im_base[:, :, 0:3] * alpha_b * (1 - alpha_a)
            ) / alpha_0
            im_base = im_combined

    return im_combined


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray, copy: bool = True) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation


def run_otsu_thresholding(data, return_type="assignments"):
    flattened_data = data.flatten()
    threshold_value = filters.threshold_otsu(flattened_data)
    
    if return_type == "threshold":
        return threshold_value
    else:
        return np.where(flattened_data <= threshold_value, 0, 1)
