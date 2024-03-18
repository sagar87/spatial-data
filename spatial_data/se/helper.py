import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

from ..constants import COLORS


def render_label(mask, cmap_mask, img=None, alpha=0.2, alpha_boundary=0, mode="inner"):
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


def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])


def label_segmentation_mask(
    segmentation: np.ndarray,
    annotation: pd.DataFrame,
    label_col: str = "type",
    cell_col: str = "id",
) -> np.ndarray:
    """
    Relabels a segmentation according to the annotations df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    cell_types = annotation.loc[:, label_col].values.astype(int)
    cell_ids = annotation.loc[:, cell_col].values

    if 0 in cell_types:
        cell_types += 1

    for t in np.unique(cell_types):
        mask = np.isin(segmentation, cell_ids[cell_types == t])
        labeled_segmentation[mask] = t

    # remove cells that are not indexed
    neg_mask = ~np.isin(segmentation, cell_ids)
    labeled_segmentation[neg_mask] = 0

    return labeled_segmentation


def label_cells(raw_image, labeled_segmentation, cmap, **kwargs):
    return render_label(labeled_segmentation, img=raw_image, cmap=cmap, **kwargs)


def generate_cmap(num_cell_types, colors=COLORS, labels=None):
    cmap = ListedColormap(colors, N=num_cell_types)
    if labels is None:
        labels = ["BG"] + [f"Cell type {i}" for i in range(num_cell_types)]

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=t, markerfacecolor=c, markersize=15)
        for c, t in zip(colors, labels)
    ]
    return cmap, legend_elements


# adapted from CellSeg
def compute_centroids(flatmasks):
    masks = flatmasks.copy()
    num_masks = len(np.unique(masks)) - 1
    indices = np.where(masks != 0)
    values = masks[indices[0], indices[1]]

    maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"x", 1:"y", 2:"id"})
    centroids = maskframe.groupby('id').agg({'x': 'mean', 'y': 'mean'}).to_records(index = False).tolist()

    return centroids


def compute_boundbox(flatmasks):
    masks = flatmasks.copy()
    num_masks = len(np.unique(masks)) - 1
    indices = np.where(masks != 0)
    values = masks[indices[0], indices[1]]

    maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"y", 1:"x", 2:"id"})
    bb_mins = maskframe.groupby('id').agg({'y': 'min', 'x': 'min'}).to_records(index = False).tolist()
    bb_maxes = maskframe.groupby('id').agg({'y': 'max', 'x': 'max'}).to_records(index = False).tolist()
    
    return bb_mins, bb_maxes

# adapted from CellSeg, but only sequential mode
def grow_masks(flatmasks, growth, bb_mins, bb_maxes):
    masks = flatmasks.copy()
    num_masks = len(np.unique(masks)) - 1
    
    print(num_masks)

    Y, X = masks.shape
    struc = disk(1)
    for _ in range(growth):
        for i in range(num_masks):
            mins = bb_mins[i]
            maxes = bb_maxes[i]
            minY, minX, maxY, maxX = mins[0] - 3*growth, mins[1] - 3*growth, maxes[0] + 3*growth, maxes[1] + 3*growth
            if minX < 0: minX = 0
            if minY < 0: minY = 0
            if maxX >= X: maxX = X - 1
            if maxY >= Y: maxY = Y - 1

            currreg = masks[minY:maxY, minX:maxX]
            mask_snippet = (currreg == i + 1)
            full_snippet = currreg > 0
            other_masks_snippet = full_snippet ^ mask_snippet
            dilated_mask = binary_dilation(mask_snippet, struc)
            final_update = (dilated_mask ^ full_snippet) ^ other_masks_snippet

            pix_to_update = np.nonzero(final_update)

            pix_X = np.array([min(j + minX, X) for j in pix_to_update[1]])
            pix_Y = np.array([min(j + minY, Y) for j in pix_to_update[0]])
            print([pix_Y, pix_X])
            
            try:
                masks[pix_Y, pix_X] = i + 1
            except IndexError:
                print(f"Skipping mask {i}")

        return masks