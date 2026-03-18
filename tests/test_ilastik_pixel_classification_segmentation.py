import shutil
from pathlib import Path

import pytest
from devtools import debug

from ilastik_tasks.ilastik_pixel_classification_segmentation import (
    ilastik_pixel_classification_segmentation,
)
from ilastik_tasks.ilastik_utils import (
    AdvancedIlastikParameters,
    IlastikChannels,
    IteratorConfig,
)


@pytest.fixture(scope="function")
def ome_zarr_3d_url(tmp_path: Path, testdata_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = testdata_path / "AssayPlate_Greiner_CELLSTAR655090_B03_0.zarr"
    dest_dir = (tmp_path / "ilastik_data_3d").as_posix()
    if Path(dest_dir).exists():
        shutil.rmtree(dest_dir)
    debug(f"Copying {source_dir} to {dest_dir}")
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_ilastik_pixel_classification_segmentation_task_3D_dual_channel(
    ome_zarr_3d_url,
):
    """
    Test the 3D ilastik_pixel_classification_segmentation task with dual channel input.
    """
    ilastik_model = (
        Path(__file__).parent / "data" / "pixel_classifier_3D_dual_channel.ilp"
    ).as_posix()

    ilastik_pixel_classification_segmentation(
        zarr_url=ome_zarr_3d_url,
        channels=IlastikChannels(mode="label", identifiers=["DAPI_2", "ECadherin_2"]),
        iterator_configuration=IteratorConfig(roi_table="FOV_ROI_table"),
        ilastik_model=str(ilastik_model),
        label_name="test_label",
        level_path="0",
        advanced_parameters=AdvancedIlastikParameters(
            foreground_class=0,
            threshold=0.5,
            min_size=15,
        ),
    )

    # Test failing of task if model was trained with two channels
    # but only one is provided
    with pytest.raises(ValueError):
        ilastik_pixel_classification_segmentation(
            zarr_url=ome_zarr_3d_url,
            channels=IlastikChannels(mode="label", identifiers=["DAPI_2"]),
            iterator_configuration=IteratorConfig(roi_table="FOV_ROI_table"),
            ilastik_model=str(ilastik_model),
            label_name="test_label",
            level_path="0",
            advanced_parameters=AdvancedIlastikParameters(
                foreground_class=0,
                threshold=0.5,
                min_size=15,
            ),
        )


def test_ilastik_pixel_classification_segmentation_task_3D_single_channel(
    ome_zarr_3d_url,
):
    """
    Test the 3D ilastik_pixel_classification_segmentation task
    with single channel input.
    """
    ilastik_model = (
        Path(__file__).parent / "data/pixel_classifier_3D_single_channel.ilp"
    ).as_posix()

    ilastik_pixel_classification_segmentation(
        zarr_url=ome_zarr_3d_url,
        channels=IlastikChannels(mode="label", identifiers=["DAPI_2"]),
        iterator_configuration=IteratorConfig(roi_table="well_ROI_table"),
        ilastik_model=str(ilastik_model),
        label_name="test_label",
        level_path="0",
        advanced_parameters=AdvancedIlastikParameters(
            foreground_class=0,
            threshold=0.5,
            min_size=15,
        ),
    )

    # Test failing of task if model was trained with one channel
    # but two are provided
    with pytest.raises(ValueError):
        ilastik_pixel_classification_segmentation(
            zarr_url=ome_zarr_3d_url,
            channels=IlastikChannels(
                mode="label", identifiers=["DAPI_2", "ECadherin_2"]
            ),
            iterator_configuration=IteratorConfig(roi_table="well_ROI_table"),
            ilastik_model=str(ilastik_model),
            label_name="test_label",
            level_path="0",
            advanced_parameters=AdvancedIlastikParameters(
                foreground_class=0,
                threshold=0.5,
                min_size=15,
            ),
        )


@pytest.fixture(scope="function")
def ome_zarr_2d_url(tmp_path: Path, testdata_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder for 2D tests.
    """
    source_dir = testdata_path / "test_mip_ome.zarr"
    dest_dir = (tmp_path / "ilastik_data_2d").as_posix()
    if Path(dest_dir).exists():
        shutil.rmtree(dest_dir)
    debug(f"Copying {source_dir} to {dest_dir}")
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_ilastik_pixel_classification_segmentation_task_2D_single_channel(
    ome_zarr_2d_url,
):
    """
    Test the 2D ilastik_pixel_classification_segmentation task
    with single channel input.
    """
    ilastik_model = (
        Path(__file__).parent / "data/pixel_classifier_2D_single_channel.ilp"
    ).as_posix()

    ilastik_pixel_classification_segmentation(
        zarr_url=ome_zarr_2d_url,
        channels=IlastikChannels(mode="label", identifiers=["DAPI"]),
        iterator_configuration=IteratorConfig(roi_table="FOV_ROI_table"),
        ilastik_model=str(ilastik_model),
        label_name="test_label",
        level_path="1",
        advanced_parameters=AdvancedIlastikParameters(
            foreground_class=0,
            threshold=0.5,
            min_size=15,
        ),
    )

    # Test failing of task if model was trained with one channel
    # but two are provided
    with pytest.raises(ValueError):
        ilastik_pixel_classification_segmentation(
            zarr_url=ome_zarr_2d_url,
            channels=IlastikChannels(mode="label", identifiers=["DAPI", "ECadherin"]),
            iterator_configuration=IteratorConfig(roi_table="FOV_ROI_table"),
            ilastik_model=str(ilastik_model),
            label_name="test_label",
            level_path="1",
            advanced_parameters=AdvancedIlastikParameters(
                foreground_class=0,
                threshold=0.5,
                min_size=15,
            ),
        )
