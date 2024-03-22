import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.neighbors import kneighbors_graph
from skimage.morphology import disk, dilation
from scipy.spatial.distance import cdist

from ..constants import COLORS


# the three methods below were adapted from CellSeg
# in some edge cases, it behaves a bit weird, and we might want to customize this method
def compute_centroids(flatmasks):
    masks = flatmasks.copy()
    num_masks = len(np.unique(masks)) - 1
    indices = np.where(masks != 0)
    values = masks[indices[0], indices[1]]

    maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(
        columns={0: "x", 1: "y", 2: "id"}
    )
    centroids = maskframe.groupby("id").agg({"x": "mean", "y": "mean"}).to_records(index=False).tolist()

    return centroids


def remove_overlaps_nearest_neighbors(masks, centroids):
    final_masks = np.max(masks, axis=2)
    collisions = np.nonzero(np.sum(masks > 0, axis=2) > 1)
    collision_masks = masks[collisions]
    collision_index = np.nonzero(collision_masks)
    collision_masks = collision_masks[collision_index]
    collision_frame = pd.DataFrame(np.transpose(np.array([collision_index[0], collision_masks]))).rename(
        columns={0: "collis_idx", 1: "mask_id"}
    )
    grouped_frame = collision_frame.groupby("collis_idx")
    for collis_idx, group in grouped_frame:
        collis_pos = np.expand_dims(np.array([collisions[0][collis_idx], collisions[1][collis_idx]]), axis=0)
        prevval = final_masks[collis_pos[0, 0], collis_pos[0, 1]]
        mask_ids = list(group["mask_id"])
        curr_centroids = np.array([centroids[mask_id - 1] for mask_id in mask_ids])
        dists = cdist(curr_centroids, collis_pos)
        closest_mask = mask_ids[np.argmin(dists)]
        final_masks[collis_pos[0, 0], collis_pos[0, 1]] = closest_mask

    return final_masks


def grow_masks(flatmasks, centroids, growth, num_neighbors=30):
    masks = flatmasks.copy()
    num_masks = len(np.unique(masks)) - 1
    indices = np.where(masks != 0)
    values = masks[indices[0], indices[1]]

    maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(
        columns={0: "x", 1: "y", 2: "id"}
    )
    cent_array = maskframe.groupby("id").agg({"x": "mean", "y": "mean"}).to_numpy()
    connectivity_matrix = kneighbors_graph(cent_array, num_neighbors).toarray() * np.arange(1, num_masks + 1)
    connectivity_matrix = connectivity_matrix.astype(int)
    labels = {}
    for n in range(num_masks):
        connections = list(connectivity_matrix[n, :])
        connections.remove(0)
        layers_used = [labels[i] for i in connections if i in labels]
        layers_used.sort()
        currlayer = 0
        for layer in layers_used:
            if currlayer != layer:
                break
            currlayer += 1
        labels[n + 1] = currlayer

    possible_layers = len(list(set(labels.values())))
    label_frame = pd.DataFrame(list(labels.items()), columns=["maskid", "layer"])
    image_h, image_w = masks.shape
    expanded_masks = np.zeros((image_h, image_w, possible_layers), dtype=np.uint32)

    grouped_frame = label_frame.groupby("layer")
    for layer, group in grouped_frame:
        currids = list(group["maskid"])
        masklocs = np.isin(masks, currids)
        expanded_masks[masklocs, layer] = masks[masklocs]

    dilation_mask = disk(1)
    grown_masks = np.copy(expanded_masks)
    for _ in range(growth):
        for i in range(possible_layers):
            grown_masks[:, :, i] = dilation(grown_masks[:, :, i], dilation_mask)
    flatmasks = remove_overlaps_nearest_neighbors(grown_masks, centroids)
    
    # assert that the new number of masks is the same as before
    assert len(np.unique(flatmasks)) - 1 == num_masks, f"Number of masks changed after growing: went from {num_masks} to {len(np.unique(flatmasks)) - 1}. Ensure that your masks are labeled 1 to n."
    return flatmasks
