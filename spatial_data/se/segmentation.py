from typing import Union

import numpy as np
import xarray as xr
from skimage.measure import regionprops_table
from tqdm import tqdm

from ..constants import Dims, Features, Layers
from .helper import grow_masks, compute_centroids

PROPS_DICT = {"centroid-1": Features.X, "centroid-0": Features.Y}


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray, copy: bool = True) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation


def _relabel_cells(segmentation: np.ndarray):
    # this method relabels cells, so if you have non-consecutive labels, they will be turned into labels from 1-n again
    # this is important since CellSeg's mask growing relies on this assumption
    unique_values = np.unique(segmentation)  # Find unique values in the array
    num_unique_values = len(unique_values)  # Get the number of unique values

    # Create a mapping from original values to new values
    value_map = {value: i for i, value in enumerate(unique_values)}

    # Map the original array to the new values using the mapping
    segmentation_relabeled = np.vectorize(lambda x: value_map[x])(segmentation)

    return segmentation_relabeled, value_map


@xr.register_dataset_accessor("se")
class SegmentationAccessor:
    """Handles everything that relates to the provided segmentation."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def filter_cells(self, min_size: int = 75, max_size: int = 300, verbose: bool = True):
        """
        Filters cells by size.
        """
                
        # checking if the segmentation layer is present
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")

        # checking if a grown segmentation mask already exists. If it does, we will use that for filtering, otherwise the normal one
        segmentation = self._obj[Layers.SEGMENTATION]

        if verbose:
            print(f"Number of cells before filtering: {len(self._obj.coords['cells'].values)}")
        self._obj = self._obj.pp.add_observations(["label", "area"]).la.filter_by_obs(
            col="area", func=lambda x: (x >= min_size) & (x <= max_size)
        )
        # these are the cells passing the filter
        cells = self._obj.coords["cells"].values
        if verbose:
            print(f"Number of cells after filtering: {len(self._obj.coords['cells'].values)}")

        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation.values, cells)
        # relabeling cells in the segmentation mask so the IDs go from 1 to n again
        segmentation, relabel_dict = _relabel_cells(segmentation)
        
        # updating the cell names of the object using the relabel dict
        self._obj = self._obj.assign_coords(cells=[relabel_dict[cell] for cell in self._obj.coords["cells"].values])

        # putting the resulting segmentation mask into a data array and adding it to the xarray object
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # removing the old segmentation
        self._obj = self._obj.drop_vars(Layers.SEGMENTATION)
        self._obj = xr.merge([self._obj, da])
        
        # running some checks to ensure consistency between the cell labels and the segmentation mask
        assert len(self._obj.coords["cells"].values) == len(np.unique(segmentation)) - 1, f"Number of cells and unique cell IDs in the segmentation mask do not match. Coords have {len(self._obj.coords['cells'].values)} and segmentation has {len(np.unique(segmentation)) - 1}."
        # making sure all cells (non-zero entries) in the segmentation mask are also in the coords
        assert np.all(np.isin(np.unique(segmentation[segmentation != 0]), self._obj.coords["cells"].values)), "Not all cells in the segmentation mask are in the coords."
        
        return self._obj

    def grow_cells(self, iterations: int = 2, verbose: bool = True):
        """
        Grows the cells in the segmentation mask.
        """
        #raise NotImplementedError(
        #    "This method is not yet implemented, because the CellSeq code has some weird behavior."
        #)
        # checking if the segmentation layer is present
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")
        
        # checking if the segmentation masks are labeled 1 to n
        unique_values = np.unique(self._obj[Layers.SEGMENTATION].values)
        assert np.all(unique_values == np.arange(0, len(unique_values))), "Segmentation mask is not labeled 1 to n, which is required for mask growing to work properly"

        segmentation = self._obj[Layers.SEGMENTATION].values
        centroids = compute_centroids(segmentation)
        num_neighbors = min(30, self._obj.dims["cells"] - 1)
        masks_grown = grow_masks(segmentation, centroids, iterations, num_neighbors=num_neighbors)

        # assigning the grown masks to the object
        da = xr.DataArray(
            masks_grown,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )
        
        # replacing the old segmentation mask with the new one
        self._obj = self._obj.drop_vars(Layers.SEGMENTATION)
        self._obj = xr.merge([self._obj, da])

        # after segmentation masks were grown, the areas need to be updated
        self._obj = self._obj.pp.add_observations(['label', 'area'], overwrite=True)
        
        return self._obj
