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
            name=Layers.SEGMENTATION,
        )
        
        # removing the old segmentation
        self._obj = self._obj.drop_vars(Layers.SEGMENTATION)
        
        return xr.merge([self._obj, da])
    
    
    def grow_cells(self, iterations: int = 2, verbose: bool = True):
        """
        Grows the cells in the segmentation mask.
        """
        raise NotImplementedError("This method is not yet implemented, because the CellSeq code has some weird behavior.")
        # checking if the segmentation layer is present
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")
        
        segmentation = self._obj[Layers.SEGMENTATION].values
        centroids = compute_centroids(segmentation)
        num_neighbors = min(30, self._obj.dims['cells'] - 1)
        print(num_neighbors)
        masks_grown = grow_masks(segmentation, centroids, iterations, num_neighbors=num_neighbors)
        
        # assigning the grown masks to the object
        da = xr.DataArray(
            masks_grown,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )
        
        # TODO: apparently we are losing cells here, this needs more work
        # after segmentation masks were grown, the areas need to be updated
        # self._obj = self._obj.pp.add_observations(['label', 'area'])
        print(len(centroids))
        print(len(np.unique(segmentation)))
        print(len(np.unique(masks_grown)))
        table = regionprops_table(masks_grown, properties=("label", "area"))
        print(table["area"].shape)
        self._obj = self._obj.pp.add_properties(table["area"], prop="area_grown")
        
        # TODO: reactivate this
        # return xr.merge([self._obj, da])        
        