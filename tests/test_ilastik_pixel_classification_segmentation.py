import shutil
from pathlib import Path

import pytest
from devtools import debug
from fractal_tasks_core.channels import ChannelInputModel

from ilastik_tasks.ilastik_pixel_classification_segmentation import (
    ilastik_pixel_classification_segmentation,
)


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = (Path(__file__).parent / "data/ngff_example/my_image").as_posix()
    dest_dir = (tmp_path / "my_image").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_ilastik_pixel_classification_segmentation_task(test_data_dir):
    """
    Test the ilastik_pixel_classification_segmentation task.
    """
    ilastik_model = (Path(__file__).parent / "data/pixel_classifier_2D.ilp").as_posix()

    ilastik_pixel_classification_segmentation(
        zarr_url=test_data_dir,
        level=0,
        channel=ChannelInputModel(label="DAPI"),
        ilastik_model=str(ilastik_model),
        output_label_name="test_label",
    )
