from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import wiener
from skimage.measure import regionprops_table
from skimage.restoration import unsupervised_wiener

from ..base_logger import logger
from ..constants import COLORS, Attrs, Dims, Features, Layers, Props
from ..la.label import _format_labels
from ..pl import _get_listed_colormap
from .intensity import mean_intensity
from .utils import (
    _colorize,
    _label_segmentation_mask,
    _normalize,
    _remove_segmentation_mask_labels,
    _remove_unlabeled_cells,
    _render_label,
    run_otsu_thresholding,
)


@xr.register_dataset_accessor("pp")
class PreprocessingAccessor:
    """The image accessor enables fast indexing and preprocesses image.data"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices) -> xr.Dataset:
        """Fast subsetting the image container. The following examples show how
        the user can subset the image container:

        Subset the image container using x and y coordinates:
        >> ds.pp[0:50, 0:50]

        Subset the image container using x and y coordinates and channels:
        >> ds.pp['Hoechst', 0:50, 0:50]

        Subset the image container using channels:
        >> ds.pp['Hoechst']

        Multiple channels can be selected by passing a list of channels:
        >> ds.pp[['Hoechst', 'CD4']]

        Parameters:
        -----------
        indices: str, slice, list, tuple
            The indices to subset the image container.
        Returns:
        --------
        xarray.Dataset
            The subsetted image container.
        """

        # argument handling
        if type(indices) is str:
            c_slice = [indices]
            x_slice = slice(None)
            y_slice = slice(None)
        elif type(indices) is slice:
            c_slice = slice(None)
            x_slice = indices
            y_slice = slice(None)
        elif type(indices) is list:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = indices
                x_slice = slice(None)
                y_slice = slice(None)
        elif type(indices) is tuple:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = [*indices]
                x_slice = slice(None)
                y_slice = slice(None)

            if len(indices) == 2:
                if (type(indices[0]) is slice) & (type(indices[1]) is slice):
                    c_slice = slice(None)
                    x_slice = indices[0]
                    y_slice = indices[1]
                elif (type(indices[0]) is str) & (type(indices[1]) is slice):
                    # Handles arguments in form of im['Hoechst', 500:1000]
                    c_slice = [indices[0]]
                    x_slice = indices[1]
                    y_slice = slice(None)
                elif (type(indices[0]) is list) & (type(indices[1]) is slice):
                    c_slice = indices[0]
                    x_slice = indices[1]
                    y_slice = slice(None)
                else:
                    raise AssertionError("Some error in handling the input arguments")

            elif len(indices) == 3:
                if type(indices[0]) is str:
                    c_slice = [indices[0]]
                elif type(indices[0]) is list:
                    c_slice = indices[0]
                else:
                    raise AssertionError("First index must index channel coordinates.")

                if (type(indices[1]) is slice) & (type(indices[2]) is slice):
                    x_slice = indices[1]
                    y_slice = indices[2]

        ds = self._obj.pp.get_channels(c_slice)
        return ds.pp.get_bbox(x_slice, y_slice)

    def get_bbox(self, x_slice: slice, y_slice: slice) -> xr.Dataset:
        """
        Returns the bounds of the image container.

        Parameters
        ----------
        x_slice : slice
            The slice representing the x-coordinates for the bounding box.
        y_slice : slice
            The slice representing the y-coordinates for the bounding box.

        Returns:
        --------
        xarray.Dataset
            The updated image container.
        """

        # get the dimensionality of the image
        xdim = self._obj.coords[Dims.X]
        ydim = self._obj.coords[Dims.Y]

        # set the start and stop indices
        x_start = xdim[0] if x_slice.start is None else x_slice.start
        y_start = ydim[0] if y_slice.start is None else y_slice.start
        x_stop = xdim[-1] if x_slice.stop is None else x_slice.stop
        y_stop = ydim[-1] if y_slice.stop is None else y_slice.stop

        # set up query
        query = {
            Dims.X: x_slice,
            Dims.Y: y_slice,
        }

        # handle case when there are cells in the image
        if Dims.CELLS in self._obj.dims:
            num_cells = self._obj.dims[Dims.CELLS]

            coords = self._obj[Layers.OBS]
            cells = (
                (coords.loc[:, Features.X] >= x_start)
                & (coords.loc[:, Features.X] <= x_stop)
                & (coords.loc[:, Features.Y] >= y_start)
                & (coords.loc[:, Features.Y] <= y_stop)
            ).values
            # calculates the number of cells that were dropped due setting the bounding box
            lost_cells = num_cells - sum(cells)

            if lost_cells > 0:
                logger.warning(f"Dropped {lost_cells} cells.")

            # finalise query
            query[Dims.CELLS] = cells

        return self._obj.sel(query)

    def get_channels(self, channels: Union[List[str], str]) -> xr.Dataset:
        """
        Returns a single channel as a numpy array.

        Parameters
        ----------
        channels: Union[str, list]
            The name of the channel or a list of channel names.

        Returns
        -------
        xarray.Dataset
            The selected channels as a new image container.
        """
        if isinstance(channels, str):
            channels = [channels]
        # build query
        query = {Dims.CHANNELS: channels}

        return self._obj.sel(query)

    def add_channel(self, channels: Union[str, list], array: np.ndarray) -> xr.Dataset:
        """
        Adds channel(s) to an existing image container.

        Parameters
        ----------
        channels : Union[str, list]
            The name of the channel or a list of channel names to be added.
        array : np.ndarray
            The numpy array representing the channel(s) to be added.

        Returns
        -------
        xarray.Dataset
            The updated image container with added channel(s).
        """
        assert type(array) is np.ndarray, "Added channel(s) must be numpy arrays"

        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        if type(channels) is str:
            channels = [channels]

        self_channels, self_x_dim, self_y_dim = self._obj[Layers.IMAGE].shape
        other_channels, other_x_dim, other_y_dim = array.shape

        assert (
            len(channels) == other_channels
        ), "The length of channels must match the number of channels in array (DxMxN)."
        assert (self_x_dim == other_x_dim) & (self_y_dim == other_y_dim), "Dims do not match."

        da = xr.DataArray(
            array,
            coords=[channels, range(other_x_dim), range(other_y_dim)],
            dims=Dims.IMAGE,
            name=Layers.IMAGE,
        )

        return xr.merge([self._obj, da])

    def add_segmentation(self, segmentation: np.ndarray, copy: bool = True) -> xr.Dataset:
        """
        Adds a segmentation mask (_segmentation) field to the xarray dataset.

        Parameters
        ----------
        segmentation : np.ndarray
            A segmentation mask, i.e., a np.ndarray with image.shape = (x, y),
            that indicates the location of each cell.
        copy : bool
            If true the segmentation mask is copied.

        Returns:
        --------
        xr.Dataset
            The amended xarray.
        """

        assert ~np.any(segmentation < 0), "A segmentation mask may not contain negative numbers."

        y_dim, x_dim = segmentation.shape

        assert (x_dim == self._obj.dims[Dims.X]) & (
            y_dim == self._obj.dims[Dims.Y]
        ), "The shape of segmentation mask does not match that of the image."

        if copy:
            segmentation = segmentation.copy()

        # crete a data array with the segmentation mask
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # add cell coordinates
        obj = self._obj.copy()
        obj.coords[Dims.CELLS] = np.unique(segmentation[segmentation > 0]).astype(int)

        return xr.merge([obj, da])

    def add_observations(
        self,
        properties: Union[str, list, tuple] = ("label", "centroid"),
        return_xarray: bool = False,
    ) -> xr.Dataset:
        """
        Adds properties derived from the mask to the image container.

        Parameters
        ----------
        properties : Union[str, list, tuple]
            A list of properties to be added to the image container. See
            skimage.measure.regionprops_table for a list of available properties.
        return_xarray : bool
            If true, the function returns an xarray.DataArray with the properties
            instead of adding them to the image container.

        Returns
        -------
        xr.DataSet
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        if type(properties) is str:
            properties = [properties]

        if "label" not in properties:
            properties = ["label", *properties]

        table = regionprops_table(self._obj[Layers.SEGMENTATION].values, properties=properties)

        label = table.pop("label")
        data = []
        cols = []

        for k, v in table.items():
            if Dims.FEATURES in self._obj.coords:
                if k in self._obj.coords[Dims.FEATURES] and not return_xarray:
                    logger.warning(f"Found {k} in _obs. Skipping.")
                    continue
            cols.append(k)
            data.append(v)

        if len(data) == 0:
            logger.warning("Warning: No properties were added.")
            return self._obj

        da = xr.DataArray(
            np.stack(data, -1),
            coords=[label, cols],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        if return_xarray:
            return da

        # if there are already observations, concatenate them
        if Layers.OBS in self._obj:
            logger.info("Found _obs in image container. Concatenating.")
            da = xr.concat(
                [self._obj[Layers.OBS].copy(), da],
                dim=Dims.FEATURES,
            )

        return xr.merge([self._obj, da])

    # TODO: channels is not used here, needs to be implemented properly
    def add_quantification(
        self,
        channels: Union[str, list] = "all",
        func=mean_intensity,
        remove_unlabeled=True,
        key_added: str = Layers.INTENSITY,
        return_xarray=False,
    ) -> xr.Dataset:
        """
        Quantify channel intensities over the segmentation mask.

        Parameters
        ----------
        channels : Union[str, list], optional
            The name of the channel or a list of channel names to be added. Default is "all".
        func : Callable, optional
            The function used for quantification. Default is mean_intensity.
        remove_unlabeled : bool, optional
            Whether to remove unlabeled cells. Default is True.
        key_added : str, optional
            The key under which the quantification data will be stored in the image container. Default is Layers.INTENSITY.
        return_xarray : bool, optional
            If True, the function returns an xarray.DataArray with the quantification data instead of adding it to the image container.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The updated image container with added quantification data or the quantification data as a separate xarray.DataArray.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        if key_added in self._obj:
            logger.warning(f"Found {key_added} in image container. Please add a different key.")
            return self._obj

        if Dims.CELLS not in self._obj.coords:
            logger.warning("No cell coordinates found. Adding _obs table.")
            self._obj = self._obj.pp.add_observations()

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        segmentation = self._obj[Layers.SEGMENTATION].values
        segmentation = _remove_unlabeled_cells(segmentation, self._obj.coords[Dims.CELLS].values)

        image = np.rollaxis(self._obj[Layers.IMAGE].values, 0, 3)
        props = regionprops_table(segmentation, intensity_image=image, extra_properties=(func,))
        cell_idx = props.pop("label")
        for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
            if k.startswith(func.__name__):
                measurements.append(props[k])

        da = xr.DataArray(
            np.stack(measurements, -1),
            coords=[cell_idx, all_channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        if return_xarray:
            return da

        return xr.merge([self._obj, da])

    def add_quantification_from_dataframe(self, df: pd.DataFrame, key_added: str = Layers.INTENSITY) -> xr.Dataset:
        """
        Adds an observation table to the image container. Columns of the
        dataframe have to match the channel coordinates of the image
        container, and the index of the dataframe has to match the cell coordinates
        of the image container.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with the quantification values.
        key_added : str, optional
            The key under which the quantification data will be added to the image container.

        Returns
        -------
        xr.DataSet
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found. A segmentation mask is required to add quantification.")

        # pulls out the cell and channel coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values
        channels = self._obj.coords[Dims.CHANNELS].values

        # create a data array from the dataframe
        da = xr.DataArray(
            df.loc[cells, channels].values,
            coords=[cells, channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        return xr.merge([self._obj, da])

    def add_marker_binarization(
        self, channels: Union[str, list] = "all", method=Features.BINARIZATION_METHODS[0], key: str = Layers.INTENSITY
    ) -> xr.Dataset:
        """
        Binarizes all channels via thresholding.
        TODO: For now it does this for all the channels, but should implement it so that it can also only perform the binarization on selected channels
        """
        # check if the method is a valid binarization method
        if method not in Features.BINARIZATION_METHODS:
            raise ValueError(
                f"Invalid binarization method {method}. Please choose one of {Features.BINARIZATION_METHODS}."
            )

        # getting the expression matrix in the form of a numpy array
        expression_matrix = self._obj[key].values
        binarization_matrix = []
        # iterating through each marker
        for marker_index in range(len(self._obj.coords[Dims.CHANNELS])):
            # getting the data as a 2D array
            data = np.array(expression_matrix[:, marker_index]).reshape(-1, 1)
            binarization_matrix.append(run_otsu_thresholding(data))
        binarization_matrix = np.array(binarization_matrix).T
        col_names = [f"{x}_binarization_{method}" for x in self._obj.coords[Dims.CHANNELS].values]

        # putting the binarization matrix in the form of a DataArray
        da = xr.DataArray(
            binarization_matrix,
            coords=[self._obj.coords[Dims.CELLS].values, col_names],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        # if there are already observations, concatenate them
        if Layers.OBS in self._obj:
            logger.info(f"Found {Layers.OBS} in image container. Concatenating.")
            da = xr.concat(
                [self._obj[Layers.OBS].copy(), da],
                dim=Dims.FEATURES,
            )

        return xr.merge([self._obj, da])

    def add_properties(
        self, array: Union[np.ndarray, list], prop: str = Features.LABELS, return_xarray: bool = False
    ) -> xr.Dataset:
        """
        Adds properties to the image container.

        Parameters
        ----------
        array : Union[np.ndarray, list]
            An array or list of properties to be added to the image container.
        prop : str, optional
            The name of the property. Default is Features.LABELS.
        return_xarray : bool, optional
            If True, the function returns an xarray.DataArray with the properties instead of adding them to the image container.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The updated image container with added properties or the properties as a separate xarray.DataArray.
        """
        unique_labels = np.unique(self._obj[Layers.OBS].sel({Dims.FEATURES: Features.LABELS}))

        if type(array) is list:
            array = np.array(array)

        if prop == Features.LABELS:
            unique_labels = np.unique(_format_labels(array))

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_labels.astype(int), [prop]],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.LABELS,
        )

        if return_xarray:
            return da

        if Layers.LABELS in self._obj:
            da = xr.concat(
                [self._obj[Layers.LABELS], da],
                dim=Dims.PROPS,
            )

        return xr.merge([da, self._obj])

    def add_labels(
        self,
        df: Union[pd.DataFrame, None] = None,
        cell_col: str = "cell",
        label_col: str = "label",
    ) -> xr.Dataset:
        """
        Adds labels to the image container.

        Parameters
        ----------
        df : Union[pd.DataFrame, None], optional
            A dataframe with the cell and label information. If None, a default labeling will be applied.
        cell_col : str, optional
            The name of the column in the dataframe representing cell coordinates. Default is "cell".
        label_col : str, optional
            The name of the column in the dataframe representing cell labels. Default is "label".
        Returns
        -------
        xr.Dataset
            The updated image container with added labels.
        """
        if df is None:
            cells = self._obj.coords[Dims.CELLS].values
            labels = np.ones(len(cells))
        else:
            cells = df.loc[:, cell_col].values.squeeze()
            labels = df.loc[:, label_col].values.squeeze()
        unique_labels = np.unique(labels)

        da = xr.DataArray(
            labels.reshape(-1, 1),
            coords=[cells, [Features.LABELS]],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        self._obj = xr.merge([self._obj.sel(cells=da.cells), da])
        return self._obj

    def transform_expression_matrix(self, key: str = Layers.INTENSITY, transform: str = "arcsinh"):
        # checking if there is an expression matrix in the data
        if key not in self._obj:
            raise ValueError(
                f"No expression matrix with key {key} found in the object. Make sure to call pp.quantify first."
            )

        if transform not in Features.TRANSFORMS:
            raise ValueError(f"Invalid transform {transform}. Please choose one of {Features.TRANSFORMS}.")

        # getting the expression matrix
        expression_matrix = self._obj[key]

        # applying the transformation
        if transform == "arcsinh":
            expression_matrix = np.arcsinh(expression_matrix)
        elif transform == "log":
            # handling negative inputs by raising a ValueError and exiting the function
            if np.any(expression_matrix < 0):
                raise ValueError("Expression matrix contains negative values. Log transformation is not possible.")
            expression_matrix = np.log(expression_matrix)
        elif transform == "sqrt":
            # handling negative inputs by raising a ValueError and exiting the function
            if np.any(expression_matrix < 0):
                raise ValueError("Expression matrix contains negative values. Sqrt transformation is not possible.")
            expression_matrix = np.sqrt(expression_matrix)
        elif transform == "zscore":
            expression_matrix = (expression_matrix - np.mean(expression_matrix)) / np.std(expression_matrix)

        # removing the old expression matrix if it exists
        if key in self._obj:
            self._obj = self._obj.drop_vars(key)

        # adding the transformed expression matrix to the object
        self._obj = self._obj.assign({key: expression_matrix})

        return self._obj

    def clip_expression_matrix(self, key: str = Layers.INTENSITY, min_perc: float = 0, max_perc: float = 99):
        """
        Clips the expression matrix to the specified percentiles.
        """
        # checking if there is an expression matrix in the data
        if key not in self._obj:
            raise ValueError(
                f"No expression matrix with key {key} found in the object. Make sure to call pp.quantify first."
            )

        # checking if the percentiles are within the range of 0 and 100
        if (min_perc < 0) or (min_perc > 100) or (max_perc < 0) or (max_perc > 100):
            raise ValueError("Percentiles must be within the range of 0 and 100.")

        # checking if the min_perc is smaller than the max_perc
        if min_perc >= max_perc:
            raise ValueError("min_perc must be smaller than max_perc.")

        # getting the expression matrix
        expression_matrix = self._obj[key]

        # clipping the expression matrix to the provided percentiles
        min_val = np.percentile(expression_matrix, min_perc)
        max_val = np.percentile(expression_matrix, max_perc)
        expression_matrix = np.clip(expression_matrix, min_val, max_val)

        # removing the old expression matrix if it exists
        if key in self._obj:
            self._obj = self._obj.drop_vars(key)

        # adding the clipped expression matrix to the object
        self._obj = self._obj.assign({key: expression_matrix})

        return self._obj

    def restore(self, method="wiener", **kwargs):
        """
        Restores the image using a specified method.

        Parameters
        ----------
        method : str, optional
            The method used for image restoration. Options are "wiener", "unsupervised_wiener", or "threshold". Default is "wiener".
        **kwargs : dict, optional
            Additional keyword arguments specific to the chosen method.

        Returns
        -------
        xr.Dataset
            The restored image container.
        """
        image_layer = self._obj[Layers.IMAGE]

        obj = self._obj.drop(Layers.IMAGE)

        if method == "wiener":
            restored = wiener(image_layer.values)
        elif method == "unsupervised_wiener":
            psf = np.ones((5, 5)) / 25
            restored, _ = unsupervised_wiener(image_layer.values.squeeze(), psf)
            restored = np.expand_dims(restored, 0)
        elif method == "threshold":
            value = kwargs.get("value", 128)
            rev_func = kwargs.get("rev_func", lambda x: x)
            restored = np.zeros_like(image_layer)
            idx = np.where(image_layer > rev_func(value))
            restored[idx] = image_layer[idx]

        normed = xr.DataArray(
            restored,
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.IMAGE,
        )
        return xr.merge([obj, normed])

    def normalize(self):
        """
        Performs a percentile normalization on each channel.

        Returns
        -------
        xr.Dataset
            The image container with the normalized image stored in Layers.PLOT.
        """
        image_layer = self._obj[Layers.IMAGE]
        normed = xr.DataArray(
            _normalize(image_layer.values),
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.PLOT,
        )

        return xr.merge([self._obj, normed])

    def colorize(
        self,
        colors: List[str] = ["C0", "C1", "C2", "C3"],
        background: str = "black",
        normalize: bool = True,
        merge: bool = True,
    ) -> xr.Dataset:
        """
        Colorizes a stack of images.

        Parameters
        ----------
        colors : List[str], optional
            A list of strings that denote the color of each channel. Default is ["C0", "C1", "C2", "C3"].
        background : str, optional
            Background color of the colorized image. Default is "black".
        normalize : bool, optional
            Normalize the image prior to colorizing it. Default is True.
        merge : True, optional
            Merge the channel dimension. Default is True.

        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """
        image_layer = self._obj[Layers.IMAGE]
        colored = _colorize(
            image_layer.values,
            colors=colors,
            background=background,
            normalize=normalize,
        )
        da = xr.DataArray(
            colored,
            coords=[
                image_layer.coords[Dims.CHANNELS],
                image_layer.coords[Dims.Y],
                image_layer.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs={Attrs.IMAGE_COLORS: {k.item(): v for k, v in zip(image_layer.coords[Dims.CHANNELS], colors)}},
        )

        if merge:
            da = da.sum(Dims.CHANNELS, keep_attrs=True)
            da.values[da.values > 1] = 1.0

        return xr.merge([self._obj, da])

    def render_segmentation(
        self,
        alpha: float = 0,
        alpha_boundary: float = 1,
        mode: str = "inner",
    ) -> xr.Dataset:
        """
        Render the segmentation layer of the data object.

        This method renders the segmentation layer of the data object and returns an updated data object
        with the rendered visualization. The rendered segmentation is represented in RGBA format.

        Parameters
        ----------
        alpha : float, optional
            The alpha value to control the transparency of the rendered segmentation. Default is 0.
        alpha_boundary : float, optional
            The alpha value for boundary pixels in the rendered segmentation. Default is 1.
        mode : str, optional
            The mode for rendering the segmentation: "inner" for internal region, "boundary" for boundary pixels.
            Default is "inner".

        Returns
        -------
        any
            The updated data object with the rendered segmentation as a new plot layer.

        Notes
        -----
        - The function extracts the segmentation layer and information about boundary pixels from the data object.
        - It applies the specified alpha values and mode to render the segmentation.
        - The rendered segmentation is represented in RGBA format and added as a new plot layer to the data object.
        """
        assert Layers.SEGMENTATION in self._obj, "Add Segmentation first."

        color_dict = {1: "white"}
        cmap = _get_listed_colormap(color_dict)
        segmentation = self._obj[Layers.SEGMENTATION].values
        segmentation = _remove_segmentation_mask_labels(segmentation, self._obj.coords[Dims.CELLS].values)
        # mask = _label_segmentation_mask(segmentation, cells_dict)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_label(
                segmentation,
                cmap,
                self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_label(
                segmentation,
                cmap,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )

        da = xr.DataArray(
            rendered,
            coords=[
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs=attrs,
        )
        return xr.merge([self._obj, da])

    def render_label(
        self, alpha: float = 0, alpha_boundary: float = 1, mode: str = "inner", override_color: Union[str, None] = None
    ) -> xr.Dataset:
        """
        Render the labeled cells in the data object.

        This method renders the labeled cells in the data object based on the label colors and segmentation.
        The rendered visualization is represented in RGBA format.

        Parameters
        ----------
        alpha : float, optional
            The alpha value to control the transparency of the rendered labels. Default is 0.
        alpha_boundary : float, optional
            The alpha value for boundary pixels in the rendered labels. Default is 1.
        mode : str, optional
            The mode for rendering the labels: "inner" for internal region, "boundary" for boundary pixels.
            Default is "inner".
        override_color : any, optional
            The color value to override the default label colors. Default is None.

        Returns
        -------
        any
            The updated data object with the rendered labeled cells as a new plot layer.

        Raises
        ------
        AssertionError
            If the data object does not contain label information. Use 'add_labels' function to add labels first.

        Notes
        -----
        - The function retrieves label colors from the data object and applies the specified alpha values and mode.
        - It renders the labeled cells based on the label colors and the segmentation layer.
        - The rendered visualization is represented in RGBA format and added as a new plot layer to the data object.
        - If 'override_color' is provided, all labels will be rendered using the specified color.
        """
        assert Layers.LABELS in self._obj, "Add labels via the add_labels function first."

        # TODO: Attribute class in constants.py
        color_dict = self._obj.la._label_to_dict(Props.COLOR, relabel=True)
        if override_color is not None:
            color_dict = {k: override_color for k in color_dict.keys()}

        cmap = _get_listed_colormap(color_dict)

        cells_dict = self._obj.la._cells_to_label(relabel=True)
        segmentation = self._obj[Layers.SEGMENTATION].values
        mask = _label_segmentation_mask(segmentation, cells_dict)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_label(
                mask,
                cmap,
                self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_label(mask, cmap, alpha=alpha, alpha_boundary=alpha_boundary, mode=mode)

        da = xr.DataArray(
            rendered,
            coords=[
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs=attrs,
        )

        return xr.merge([self._obj, da])
