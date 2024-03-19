from typing import List, Union
import xarray as xr
from stardist.models import StarDist2D
import csbdeep.utils
import pandas as pd
import numpy as np
from ..constants import Layers


@xr.register_dataset_accessor("ext")
class ExternalAccessor:
    """The external accessor enables the application of external tools such as StarDist or Astir"""
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def stardist(self, scale: float = 3, n_tiles: int = 12, normalize: bool = True, nuclear_channel: str = "DAPI", **kwargs) -> xr.Dataset:
        """Apply StarDist to the image"""
        
        if Layers.SEGMENTATION in self._obj:
            raise ValueError("The object already contains a segmentation mask. StarDist will not be executed.")
        
        # getting the nuclear image
        nuclear_img = self._obj.pp['DAPI'].to_array().values.squeeze()
        
        if normalize:
            nuclear_img = csbdeep.utils.normalize(nuclear_img)
        
        # Load the StarDist model
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        
        # Predict the label image
        print("Predicting")
        labels, _ = model.predict_instances(nuclear_img, scale=scale, n_tiles=(n_tiles, n_tiles))
        
        # Adding the segmentation mask to the xarray dataset
        print("Adding to object")
        return self._obj.pp.add_segmentation(labels)