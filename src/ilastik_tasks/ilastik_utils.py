"""Helper functions for ilastik tasks.

Some are moodified from
https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task/blob/main/src/fractal_cellpose_sam_task/utils.py
"""

import logging
from typing import Literal, Optional

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MaskingConfiguration(BaseModel):
    """Masking configuration.

    Args:
        mode (Literal["Table Name", "Label Name"]): Mode of masking to be applied.
            If "Table Name", the identifier refers to a masking table name.
            If "Label Name", the identifier refers to a label image name.
        identifier (str): Name of the masking table or label image
            depending on the mode.
    """

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    identifier: Optional[str] = None


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration.

    Args:
        masking (Optional[MaskingIterator]): If set, the segmentation will be
            performed only within the confines of the specified mask. A mask can be
            specified either by a label image or a Masking ROI table.
        roi_table (Optional[str]): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    masking: Optional[MaskingConfiguration] = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: Optional[str] = Field(default=None, title="Iterate Over ROIs")


class IlastikChannels(BaseModel):
    """Ilastik channels configuration.

    Args:
        This model is used to select a channel by label, wavelength ID, or index.

    Args:
        identifiers (str): Unique identifier for the channel.
            This can be a channel label, wavelength ID, or index.
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer). At least one and at most three
            identifiers must be provided.

    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifiers: list[str] = Field(default_factory=list, min_length=1, max_length=3)

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
    threshold: float = 0.0
    min_size: int = 15


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
