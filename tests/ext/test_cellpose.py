import pytest

from spatial_data.constants import Layers


def test_cellpose_subset_of_markers(dataset_full):
    dataset = dataset_full.pp[["Hoechst", "CD4", "CD8"]].pp[250:300, 250:300].pp.drop_layers(Layers.SEGMENTATION)
    with pytest.raises(
        AssertionError,
        match="You are trying to segment only a subset of the available channels.",
    ):
        dataset.ext.cellpose(channels=["Hoechst", "CD8"], gpu=False)


def test_cellpose_segmentation_already_exists(dataset_full):
    with pytest.raises(
        AssertionError,
        match=f"A segmentation mask with the key {Layers.SEGMENTATION} already exists.",
    ):
        dataset_full.ext.cellpose()
