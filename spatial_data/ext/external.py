from typing import List, Union
import xarray as xr
from stardist.models import StarDist2D
import csbdeep.utils
import pandas as pd
import numpy as np
import warnings
import astir
import torch
from ..constants import Layers, Dims


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
        labels, _ = model.predict_instances(nuclear_img, scale=scale, n_tiles=(n_tiles, n_tiles))
        
        # Adding the segmentation mask to the xarray dataset
        return self._obj.pp.add_segmentation(labels)
    
    
    def astir(self, marker_dict: dict, key: str = Layers.INTENSITY, threshold: float = 0, seed: int = 42, learning_rate: float = 0.001, batch_size: float = 64, n_init: int = 5, 
              n_init_epochs: int = 5, max_epochs: int = 500, **kwargs):
        # warn the user if the image is of anything other than uint8
        # right now we can't do anything about that, should be implemented later
        if self._obj[Layers.IMAGE].dtype != "uint8":
            warnings.warn("The image is not of type uint8, which is required for astir to work properly.")

        expression_df = pd.DataFrame(self._obj[key].values, columns = self._obj.coords[Dims.CHANNELS].values)
        expression_df.index = self._obj.coords[Dims.CELLS].values
        model = astir.Astir(expression_df, marker_dict, dtype=torch.float64, random_seed=seed)
        model.fit_type(learning_rate=learning_rate, batch_size=batch_size, n_init=n_init, n_init_epochs=n_init_epochs, max_epochs=max_epochs)
        assigned_cell_types = model.get_celltypes(threshold=threshold)
        # assign the index to its own column (called cell)
        assigned_cell_types = assigned_cell_types.reset_index()
        # renaming the columns
        assigned_cell_types.columns = ['cell', 'label']
        # setting the cell dtype to int
        assigned_cell_types['cell'] = assigned_cell_types['cell'].astype(int)
        return self._obj.pp.add_labels(assigned_cell_types)