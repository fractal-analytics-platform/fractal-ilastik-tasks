"""Helper functions for ilastik tasks.

Some are moodified from
https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task/blob/main/src/fractal_cellpose_sam_task/utils.py
"""

import logging
from typing import Literal

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MaskingConfig(BaseModel):
    """Masking configuration."""

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    """
    Mode of masking to be applied.
        - If "Table Name", the identifier refers to a masking table name.
        - If "Label Name", the identifier refers to a label image name.
    """
    identifier: str | None = None
    """
    Name of the masking table or label image depending on the mode.
    """


class IteratorConfig(BaseModel):
    """Advanced iterator configuration."""

    masking: MaskingConfig | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    """
    If set, the segmentation will be performed only within the confines of
    the specified mask. A mask can be specified either by a label image or a
    Masking ROI table.
    """
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
    """
    Name of a ROI table. If set, the segmentation will be applied to each ROI
    in the table individually. This option can be combined with masking.
    """


class IlastikChannels(BaseModel):
    """Ilastik channels configuration.

    This model is used to select a channel by label, wavelength ID, or index.

    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    """
    Specifies how to interpret the identifiers. Can be "label", "wavelength_id", or
    "index" (must be an integer).
    """
    identifiers: list[str] = Field(min_length=1, max_length=3)
    """
    Unique identifiers for the channels. This can be channel labels, wavelength IDs, or
    indices, depending on the mode.
    At least one and at most three identifiers must be provided.
    """

    def to_list(self) -> list[ChannelSelectionModel]:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return [
            ChannelSelectionModel(identifier=identifier, mode=self.mode)
            for identifier in self.identifiers
        ]


class AdvancedIlastikParameters(BaseModel):
    """Advanced Ilastik Parameters

    Attributes:
        foreground_class (int, optional): Class to be considered as foreground
            during prediction thresholding. Defaults to 0.
        threshold (float, optional): all pixels with
            value above threshold kept for masks, decrease to find more and
            larger masks. Defaults to 0.0.
        min_size (int, optional): all ROIs below this size,
            in pixels, will be discarded. Defaults to 15.

    """

    foreground_class: int = 0
    """
    Class to be considered as foreground during prediction thresholding.
    """
    threshold: float = 0.0
    """
    All pixels with value above threshold kept for masks, decrease to find more and
    larger masks.
    """
    min_size: int = 15
    """
    All segmented objects below this size, in pixels, will be discarded.
    """


def get_expected_number_of_channels(shell) -> int:
    """Get the expected number of channels from the trained ilastik model"""
    opPixelClassification = shell.workflow.pcApplet.topLevelOperator
    len_input_images = len(opPixelClassification.InputImages)
    channel_number = []
    for i in range(len_input_images):
        channel = opPixelClassification.InputImages[i].meta.getTaggedShape()["c"]
        channel_number.append(channel)

    if len(set(channel_number)) != 1:
        raise ValueError("Inconsistent number of channels across input images.")

    return channel_number[0]
