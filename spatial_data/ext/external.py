from typing import List, Union
import xarray as xr
import pandas as pd
import numpy as np
from ..constants import Layers, Dims


@xr.register_dataset_accessor("ext")
class ExternalAccessor:
    """The external accessor enables the application of external tools such as StarDist or Astir"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def stardist(
        self,
        scale: float = 3,
        n_tiles: int = 12,
        normalize: bool = True,
        nuclear_channel: str = "DAPI",
        predict_big: bool = False,
        **kwargs
    ) -> xr.Dataset:
        """Apply StarDist to the image"""
        from stardist.models import StarDist2D
        import csbdeep.utils

        if Layers.SEGMENTATION in self._obj:
            raise ValueError("The object already contains a segmentation mask. StarDist will not be executed.")

        # getting the nuclear image
        nuclear_img = self._obj.pp[nuclear_channel].to_array().values.squeeze()

        # normalizing the image
        if normalize:
            nuclear_img = csbdeep.utils.normalize(nuclear_img)

        # Load the StarDist model
        model = StarDist2D.from_pretrained("2D_versatile_fluo")

        # Predict the label image (different methods for large or small images, see the StarDist documentation for more details)
        if predict_big:
            labels, _ = model.predict_instances_big(
                nuclear_img,
                scale=scale,
                **kwargs
            )
        else:
            labels, _ = model.predict_instances(nuclear_img, scale=scale, n_tiles=(n_tiles, n_tiles), **kwargs)

        # Adding the segmentation mask to the xarray dataset
        self._obj = self._obj.pp.add_segmentation(labels)

        # adding centroids to obs
        return self._obj.pp.add_observations()

    def astir(
        self,
        marker_dict: dict,
        key: str = Layers.INTENSITY,
        threshold: float = 0,
        seed: int = 42,
        learning_rate: float = 0.001,
        batch_size: float = 64,
        n_init: int = 5,
        n_init_epochs: int = 5,
        max_epochs: int = 500,
        cell_id_col: str = "cell_id",
        cell_type_col: str = "cell_type",
        **kwargs
    ):
        import astir
        import torch

        # raise an error if the image is of anything other than uint8
        if self._obj[Layers.IMAGE].dtype != "uint8":
            raise ValueError("The image is not of type uint8, which is required for astir to work properly. Use the dtype argument in add_quantification() to convert the image to uint8.")

        # converting the xarray to a pandas dataframe to keep track of channel names and indicies after running astir
        expression_df = pd.DataFrame(self._obj[key].values, columns=self._obj.coords[Dims.CHANNELS].values)
        expression_df.index = self._obj.coords[Dims.CELLS].values
        
        # running astir
        model = astir.Astir(expression_df, marker_dict, dtype=torch.float64, random_seed=seed)
        model.fit_type(
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_init=n_init,
            n_init_epochs=n_init_epochs,
            max_epochs=max_epochs
            **kwargs
        )
        
        # getting the predictions
        assigned_cell_types = model.get_celltypes(threshold=threshold)
        # assign the index to its own column (called cell)
        assigned_cell_types = assigned_cell_types.reset_index()
        # renaming the columns
        assigned_cell_types.columns = [cell_id_col, cell_type_col]
        # setting the cell dtype to int
        assigned_cell_types[cell_id_col] = assigned_cell_types[cell_id_col].astype(int)
        
        # adding the labels to the obs slot
        return self._obj.pp.add_labels(assigned_cell_types, cell_col=cell_id_col, label_col=cell_type_col)

    def export_to_anndata(self):
        """Export the xarray to an anndata object"""
        import anndata
        # constructing the anndata object: X is the intensity values, var_names are the channel names, and obs is the cell type predictions, sizes, etc.
        adata = anndata.AnnData(self._obj[Layers.INTENSITY].values)
        adata.var_names = self._obj.coords[Dims.CHANNELS].values
        adata.obs = pd.DataFrame(self._obj[Layers.OBS], columns=self._obj.coords[Dims.FEATURES])
        return adata

    def export_to_spatialdata(self):
        """Export the xarray to a spatial data object"""
        raise NotImplementedError(
            "This method is not yet implemented because spatialdata is not playing nice with xarray. Make sure you sort out the dependencies first before using this method."
        )
        import spatialdata
        image_raw = self._obj[Layers.IMAGE].values
        segmentation_masks = self._obj[Layers.SEGMENTATION].values
        # constructing the anndata object
        adata = anndata.AnnData(self._obj[Layers.INTENSITY].values)
        adata.var_names = self._obj.coords[Dims.CHANNELS].values
        # adding the cell type predictions, sizes, etc.
        obs_df = pd.DataFrame(self._obj[Layers.OBS], columns=self._obj.coords[Dims.FEATURES])
        adata.obs = obs_df
        spatial_data_object = spatialdata.SpatialData(
            images={"image_raw": image_raw}, labels={"segmentation_masks": segmentation_masks}, table=adata
        )
        return spatial_data_object
