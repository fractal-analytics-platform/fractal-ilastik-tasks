"""Ilastik-based segmentation task for Fractal.

Code adapted from: https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/tasks/cellpose_segmentation.py

Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>
    Joel Lüthi  <joel.luethi@fmi.ch>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of Zurich.

Ilastik adaptation by:
    Lorenzo Cerrone <lorenzo.cerrone@uzh.ch>
    Alexa McIntyre <alexa.mcintyre@uzh.ch>
    Ruth Hornbachner <ruth.hornbachner@uzh.ch>
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import platformdirs
import vigra
from ilastik import app
from ilastik.applets.dataSelection.opDataSelection import (
    PreloadedArrayDatasetInfo,
)
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage
from pydantic import Field, validate_call
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes

from ilastik_tasks.ilastik_utils import (
    AdvancedIlastikParameters,
    IlastikChannels,
    IteratorConfig,
    MaskingConfig,
    get_expected_number_of_channels,
)


def setup_ilastik(model_path: str):
    """Setup Ilastik headless shell."""
    args, _ = app.parse_known_args(
        args=[
            "--headless",
            f"--project={model_path}",
            "--readonly",
        ]
    )
    shell = app.main(args)
    return shell


def segmentation_function(
    image_data: np.ndarray,
    ilastik_model: str,
    advanced_parameters: AdvancedIlastikParameters,
) -> np.ndarray:
    """Wrap Ilastik segmentation call.

    Args:
        image_data (np.ndarray): Input image data.
        ilastik_model: Path to Ilastik model.
        advanced_parameters (AdvancedIlastikParameters): Advanced parameters for
            Ilastik segmentation.

    Returns:
        np.ndarray: Segmented image. Shape (z, y, x).
    """
    # Run ilastik headless
    # Shape from (czyx or cyx) to (tzyxc)
    if len(image_data.shape) == 3:
        image_data = np.expand_dims(image_data, axis=0)
    image_data = np.moveaxis(image_data, 0, -1)
    image_data = np.expand_dims(image_data, axis=0)
    logging.info(f"{image_data.shape=}")
    data = [
        {
            "Raw Data": PreloadedArrayDatasetInfo(
                preloaded_array=image_data, axistags=vigra.defaultAxistags("tzyxc")
            )
        }
    ]

    # Had to move Shell setup here to avoid cache build-up that lead to strange failures
    # on larger datasets
    shell = setup_ilastik_with_retries(ilastik_model)
    shell.workflow.trainable = False
    logging.info(f"Training allowed: {getattr(shell.workflow, 'trainable', 'unknown')}")

    ilastik_output = shell.workflow.batchProcessingApplet.run_export(
        data, export_to_array=True
    )[0]
    logging.info(f"{ilastik_output.shape=} after ilastik prediction")

    # Get foreground class and reshape to 3D
    ilastik_output = np.squeeze(
        ilastik_output[..., advanced_parameters.foreground_class]
    )
    logging.info(f"{ilastik_output.shape=} after foreground class selection")

    # take mask of regions above threshold
    ilastik_labels = ilastik_output > advanced_parameters.threshold

    # remove small holes
    min_size = advanced_parameters.min_size
    ilastik_labels = remove_small_holes(ilastik_labels, area_threshold=min_size)

    # label image
    ilastik_labels = label(ilastik_labels)

    # remove objects below min_size - also removes anything with major or minor axis
    # length of 0 for compatibility with current measurements task (01.24)
    if min_size > 0:
        label_props = regionprops(ilastik_labels)
        labels2remove = [
            label_props[i].label
            for i in range(ilastik_labels.max())
            if (label_props[i].area < min_size)
            or (label_props[i].axis_major_length < 1)
            or (label_props[i].major_axis_length < 1)
        ]
        logging.info(
            f"number of labels before filtering for size = {ilastik_labels.max()}"
        )
        ilastik_labels[np.isin(ilastik_labels, labels2remove)] = 0
        ilastik_labels = label(ilastik_labels)
        logging.info(
            f"number of labels after filtering for size = {ilastik_labels.max()}"
        )
        label_props = regionprops(ilastik_labels)

    ilastik_labels = np.expand_dims(ilastik_labels, axis=0).astype(np.uint32)
    return ilastik_labels


def setup_ilastik_with_retries(ilastik_model: str):
    """Setup Ilastik headless shell with retries to avoid initialization issues.

    See #17 for context

    """
    max_retries = 5
    current_round = 0
    while current_round < max_retries:
        try:
            shell = setup_ilastik(ilastik_model)
            return shell
        except FileNotFoundError:
            current_round += 1
            logging.warning(
                f"Ilastik initialization failed, retrying {current_round=}/"
                f"{max_retries}"
            )
            sleep_time = 2 ** (current_round + 1)
            time.sleep(sleep_time)

    raise FileNotFoundError(
        f"Ilastik initialization failed for model {ilastik_model} after "
        f"{max_retries} retries."
    )


def load_masked_image(
    ome_zarr,
    masking_configuration: MaskingConfig,
    level_path: Optional[str] = None,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfig): Configuration for masking.
        level_path (Optional[str]): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logging.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_image


@validate_call
def ilastik_pixel_classification_segmentation(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channels: IlastikChannels,
    label_name: str | None = None,
    level_path: str | None = None,
    # Iteration parameters
    iterator_configuration: IteratorConfig,
    # Ilastik-related parameters
    ilastik_model: str,
    advanced_parameters: AdvancedIlastikParameters = Field(
        default_factory=AdvancedIlastikParameters,
    ),
    write_roi_table: bool = True,
    overwrite: bool = True,
) -> None:
    """Run Ilastik Pixel Classification on a Zarr image.

    For more information, see:
        https://www.ilastik.org/documentation/pixelclassification/pixelclassification

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channels (IlastikChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        label_name (Optional[str]): Name of the resulting label image. If not provided,
            it will be set to "<channel_identifier>_segmented".
        level_path (Optional[str]): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (Optional[IteratorConfig]): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        ilastik_model: Path to the Ilastik model (e.g. `"somemodel.ilp"`).
        advanced_parameters (AdvancedIlastikParameters): Advanced parameters
            for Ilastik segmentation.
        write_roi_table (bool): Whether to write a masking ROI table for the segmented
            object. Defaults to True.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

    # Preliminary checks on Ilastik model
    if not os.path.exists(ilastik_model):
        raise ValueError(f"{ilastik_model=} path does not exist.")

    # Setup Ilastik headless shell
    shell = setup_ilastik_with_retries(ilastik_model)
    shell.workflow.trainable = False

    # Check if channel input fits expected number of channels of model
    expected_num_channels = get_expected_number_of_channels(shell)
    num_channels = len(channels.identifiers)

    if expected_num_channels == 2 and num_channels != 2:
        raise ValueError(
            "Ilastik model expects 2 channels as input, "
            f"but {num_channels} channel(s) were provided."
        )
    elif expected_num_channels == 1 and num_channels == 2:
        raise ValueError(
            "Ilastik model expects 1 channel as input, but 2 channels were provided."
        )
    elif expected_num_channels > 2:
        raise NotImplementedError(
            f"Expected {expected_num_channels} channels, "
            "but support for more than 2 channels is not implemented."
        )

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logging.info(f"{ome_zarr=}")

    if label_name is None:
        label_name = f"{channels.identifiers[0]}_segmented"
    label = ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    logging.info(f"Output label image: {label=}")

    if iterator_configuration is None:
        iterator_configuration = IteratorConfig()

    # Determine if we are doing 3D segmentation
    if ome_zarr.is_3d:
        axes_order = "czyx"
        pix_size_xy = label.pixel_size.yx
        assert pix_size_xy[0] == pix_size_xy[1], "Non-isotropic pixel size in XY"
    else:
        axes_order = "cyx"

    # Set up the appropriate iterator based on the configuration
    label = ome_zarr.get_label(name=label_name, path=level_path)

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        image = ome_zarr.get_image(path=level_path)
        logging.info(f"{image=}")
        iterator = SegmentationIterator(
            input_image=image,
            output_label=label,
            channel_selection=channels.to_list(),
            axes_order=axes_order,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        logging.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedSegmentationIterator(
            input_image=masked_image,
            output_label=label,
            channel_selection=channels.to_list(),
            axes_order=axes_order,
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logging.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logging.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logging.info(f"Iterator updated with ROI table: {iterator=}")

    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    logging.info("Starting processing...")
    for image_data, writer in iterator.iter_as_numpy():
        label_img = segmentation_function(
            image_data=image_data,
            ilastik_model=ilastik_model,
            advanced_parameters=advanced_parameters,
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = max(max_label, label_img.max())
        writer(label_img)

    logging.info(f"label {label_name} successfully created at {zarr_url}")

    # Optionally, write a masking ROI table for the segmented objects
    if write_roi_table:
        logging.info(f"Writing masking ROI table for label {label_name}")
        masking_table = ome_zarr.build_masking_roi_table(label_name)
        ome_zarr.add_table(
            f"{label_name}_ROI_table", masking_table, overwrite=overwrite
        )

    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=ilastik_pixel_classification_segmentation,
    )
