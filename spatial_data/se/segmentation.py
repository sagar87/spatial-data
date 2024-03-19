from typing import Union

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay
from skimage.measure import regionprops_table
from tqdm import tqdm

from ..constants import Dims, Features, Layers
from .helper import sum_intensity, grow_masks, compute_boundbox

PROPS_DICT = {"centroid-1": Features.X, "centroid-0": Features.Y}


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray, copy: bool = True) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation


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
        if Layers.SEGMENTATION_LAYERS[0] not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")
        # checking if the filtered segmentation layer already exists
        if Layers.SEGMENTATION_LAYERS[1] in self._obj:
            raise ValueError("The object already contains a filtered segmentation mask.")
        # checking if a grown segmentation mask already exists. If it does, we will use that for filtering, otherwise the normal one
        segmentation = self._obj[Layers.SEGMENTATION_LAYERS[0]]
        if Layers.SEGMENTATION_LAYERS[2] in self._obj:
            if verbose:
                print(f"Using {Layers.SEGMENTATION_LAYERS[2]} as basis for filtering")
            segmentation = self._obj[Layers.SEGMENTATION_LAYERS[2]]
        else:
            if verbose:
                print(f"Using {Layers.SEGMENTATION_LAYERS[0]} as basis for filtering")
            
        
        if verbose:
            print(f"Number of cells before filtering: {len(self._obj.coords['cells'].values)}")
        self._obj = self._obj.pp.add_observations(['label', 'area']).la.filter_by_obs(col='area', func=lambda x: (x >= min_size) & (x <= max_size))
        # these are the cells passing the filter
        cells = self._obj.coords["cells"].values
        if verbose:
            print(f"Number of cells after filtering: {len(self._obj.coords['cells'].values)}")
                
        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation.values, cells)
        
        # putting the resulting segmentation mask into a data array and adding it to the xarray object
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=f"{Layers.SEGMENTATION}_filtered",
        )
        
        return xr.merge([self._obj, da])
    
    
    def grow_cells(self, iterations: int = 2, verbose: bool = True):
        """
        Grows the cells in the segmentation mask.
        """
        # checking if the segmentation layer is present
        if Layers.SEGMENTATION_LAYERS[0] not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")
        segmentation = self._obj[Layers.SEGMENTATION_LAYERS[0]].values
        # checking if the filtered segmentation layer already exists
        if Layers.SEGMENTATION_LAYERS[1] in self._obj:
            segmentation = self._obj[Layers.SEGMENTATION_LAYERS[1]].values
            if verbose:
                print(f"Using {Layers.SEGMENTATION_LAYERS[1]} as basis for growing")
        else:
            if verbose:
                print(f"Using {Layers.SEGMENTATION_LAYERS[0]} as basis for growing")
                
        # TODO: implement logic for mask growing
        
        raise NotImplementedError("This function is not fully implemented yet.")
        
        