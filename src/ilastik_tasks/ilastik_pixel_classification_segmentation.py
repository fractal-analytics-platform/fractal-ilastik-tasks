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

import json
import logging
import os
import time
from typing import Any, Optional

import anndata as ad
import dask.array as da
import fractal_tasks_core
import numpy as np
import vigra
import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.masked_loading import masked_loading_wrapper
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_ROI_table_to_indices,
    create_roi_table_from_df_list,
    find_overlaps_in_ROI_indices,
    get_overlapping_pairs_3D,
    is_ROI_table_valid,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.utils import rescale_datasets
from ilastik import app
from ilastik.applets.dataSelection.opDataSelection import (
    PreloadedArrayDatasetInfo,
)
from pydantic import Field, validate_call
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes

from ilastik_tasks.ilastik_utils import (
    IlastikChannel1InputModel,
    IlastikChannel2InputModel,
    get_expected_number_of_channels,
)

logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def setup_ilastik(model_path: str):
    """Setup Ilastik headless shell."""
    args = app.parse_args([])
    args.headless = True
    args.project = model_path
    args.readonly = True
    shell = app.main(args)
    return shell


def segment_ROI(
    input_data: np.ndarray,
    num_labels_tot: dict[str, int],
    shell: Any,
    foreground_class: int = 0,
    threshold: float = 0.5,
    min_size: int = 15,
    label_dtype: Optional[np.dtype] = None,
    relabeling: bool = True,
) -> np.ndarray:
    """Run the Ilastik model on a single ROI.

    Args:
        input_data: np.ndarray of shape (t, z, y, x, c).
        num_labels_tot: Number of labels already in total image. Used for
            relabeling purposes. Using a dict to have a mutable object that
            can be edited from within the function without having to be passed
            back through the masked_loading_wrapper.
        shell: Ilastik headless shell.
        foreground_class: Class to be considered as foreground
            during prediction thresholding.
        threshold: Threshold for the Ilastik model.
        min_size: Minimum size for the Ilastik model.
        label_dtype: Label images are cast into this `np.dtype`.
        relabeling: Whether relabeling based on num_labels_tot is performed.

    Returns:
        np.ndarray: Segmented image. Shape (z, y, x).
    """
    # Run ilastik headless
    # Shape from (czyx) to (tzyxc)
    input_data = np.moveaxis(input_data, 0, -1)
    input_data = np.expand_dims(input_data, axis=0)
    logger.info(f"{input_data.shape=}")
    data = [
        {
            "Raw Data": PreloadedArrayDatasetInfo(
                preloaded_array=input_data, axistags=vigra.defaultAxistags("tzyxc")
            )
        }
    ]

    ilastik_output = shell.workflow.batchProcessingApplet.run_export(
        data, export_to_array=True
    )[0]
    logger.info(f"{ilastik_output.shape=} after ilastik prediction")

    # Get foreground class and reshape to 3D
    ilastik_output = np.squeeze(ilastik_output[..., foreground_class])
    logger.info(f"{ilastik_output.shape=} after foreground class selection")

    # take mask of regions above threshold
    ilastik_labels = ilastik_output > threshold

    # remove small holes
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
        logger.info(
            f"number of labels before filtering for size = {ilastik_labels.max()}"
        )
        ilastik_labels[np.isin(ilastik_labels, labels2remove)] = 0
        ilastik_labels = label(ilastik_labels)
        logger.info(
            f"number of labels after filtering for size = {ilastik_labels.max()}"
        )
        label_props = regionprops(ilastik_labels)

    # Shift labels and update relabeling counters
    if relabeling:
        num_labels_roi = np.max(ilastik_labels)
        ilastik_labels[ilastik_labels > 0] += num_labels_tot["num_labels_tot"]
        num_labels_tot["num_labels_tot"] += num_labels_roi

        # Write some logs
        logger.info(f"ROI had {num_labels_roi=}, {num_labels_tot=}")

        # Check that total number of labels is under control
        if num_labels_tot["num_labels_tot"] > np.iinfo(label_dtype).max:
            raise ValueError(
                "ERROR in re-labeling:"
                f"Reached {num_labels_tot} labels, "
                f"but dtype={label_dtype}"
            )

    return ilastik_labels.astype(label_dtype)

def setup_ilastik_with_retries(ilastik_model: str):
    """
    Setup Ilastik headless shell with retries to avoid initialization issues.
    
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
            logger.warning(
                f"Ilastik initialization failed, retrying {current_round=}/"
                f"{max_retries}"
            )
            sleep_time = 2 ** (current_round + 1)
            time.sleep(sleep_time)
    
    raise FileNotFoundError(
        f"Ilastik initialization failed for model {ilastik_model} after "
        f"{max_retries}retries."
    )


@validate_call
def ilastik_pixel_classification_segmentation(
    *,
    # Fractal parameters
    zarr_url: str,
    # Task-specific arguments
    level: int,
    channel: IlastikChannel1InputModel,
    channel2: IlastikChannel2InputModel = Field(
        default_factory=IlastikChannel2InputModel
    ),
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    use_masks: bool = True,
    # Ilastik-related arguments
    ilastik_model: str,
    relabeling: bool = True,
    foreground_class: int = 0,
    threshold: float = 0.5,
    min_size: int = 15,
    overwrite: bool = True,
) -> None:
    """Run Ilastik Pixel Classification on a Zarr image.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
                (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        channel: Primary channel for pixel classification; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`).
        channel2: Second channel for pixel classification (in the same format as
            `channel`). Use only if second channel has also been used during
            Ilastik model training.
        input_ROI_table: Name of the ROI table over which the task loops to
            apply Cellpose segmentation. Examples: `FOV_ROI_table` => loop over
            the field of views, `organoid_ROI_table` => loop over the organoid
            ROI table (generated by another task), `well_ROI_table` => process
            the whole well as one image.
        output_ROI_table: If provided, a ROI table with that name is created,
            which will contain the bounding boxes of the newly segmented
            labels. ROI tables should have `ROI` in their name.
        output_label_name: Name of the output label image (e.g. `"embryo"`).
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `emb_ROI_table`).
        ilastik_model: Path to the Ilastik model (e.g. `"somemodel.ilp"`).
        relabeling: If `True`, apply relabeling so that label values are
            unique for all objects in the well.
        foreground_class: Class to be considered as foreground during
            prediction thresholding.
        threshold: Probabiltiy threshold for the Ilastik model.
        min_size: Minimum size of the segmented objects (in pixels).
        overwrite: If `True`, overwrite the task output.
    """
    logger.info(f"Processing {zarr_url=}")

    # Preliminary checks on Cellpose model
    if not os.path.exists(ilastik_model):
        raise ValueError(f"{ilastik_model=} path does not exist.")

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}")
    logger.info(
        f"NGFF image has level-{level} pixel sizes " f"{actual_res_pxl_sizes_zyx}"
    )

    # Setup Ilastik headless shell
    shell = setup_ilastik_with_retries(ilastik_model)

    # Check if channel input fits expected number of channels of model
    expected_num_channels = get_expected_number_of_channels(shell)

    if expected_num_channels == 2 and not channel2.is_set():
        raise ValueError(
            "Ilastik model expects two channels as "
            "input but only one channel was provided"
        )
    elif expected_num_channels == 1 and channel2.is_set():
        raise ValueError(
            "Ilastik model expects 1 channel as " "input but two channels were provided"
        )

    elif expected_num_channels > 2:
        raise NotImplementedError(
            f"Expected {expected_num_channels} channels, "
            "but a maximum of channels are currently supported."
        )

    # Find channel index
    omero_channel = channel.get_omero_channel(zarr_url)
    if omero_channel:
        ind_channel = omero_channel.index
    else:
        return

    # Find channel index for second channel, if one is provided
    if channel2.is_set():
        omero_channel_2 = channel2.get_omero_channel(zarr_url)
        if omero_channel_2:
            ind_channel_c2 = omero_channel_2.index
        else:
            return

    # Set channel label
    if output_label_name is None:
        try:
            channel_label = omero_channel.label
            output_label_name = f"label_{channel_label}"
        except (KeyError, IndexError):
            output_label_name = f"label_{ind_channel}"

    # Load ZYX data
    data_zyx = da.from_zarr(f"{zarr_url}/{level}")[ind_channel]
    logger.info(f"{data_zyx.shape=}")
    if channel2.is_set():
        data_zyx_c2 = da.from_zarr(f"{zarr_url}/{level}")[ind_channel_c2]
        logger.info(f"Second channel: {data_zyx_c2.shape=}")

    # Read ROI table
    ROI_table_path = f"{zarr_url}/tables/{input_ROI_table}"
    ROI_table = ad.read_zarr(ROI_table_path)

    # Perform some checks on the ROI table
    valid_ROI_table = is_ROI_table_valid(table_path=ROI_table_path, use_masks=use_masks)
    if use_masks and not valid_ROI_table:
        logger.info(
            f"ROI table at {ROI_table_path} cannot be used for masked "
            "loading. Set use_masks=False."
        )
        use_masks = False
    logger.info(f"{use_masks=}")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # If we are not planning to use masked loading, fail for overlapping ROIs
    if not use_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {input_ROI_table} table have "
                "overlaps, but we are not using masked loading."
            )

    # Load zattrs file
    zattrs_file = f"{zarr_url}/.zattrs"
    with open(zattrs_file) as jsonfile:
        zattrs = json.load(jsonfile)

    # Preliminary checks on multiscales
    multiscales = zattrs["multiscales"]
    if len(multiscales) > 1:
        raise NotImplementedError(
            f"Found {len(multiscales)} multiscales, "
            "but only one is currently supported."
        )
    if "coordinateTransformations" in multiscales[0].keys():
        raise NotImplementedError(
            "global coordinateTransformations at the multiscales "
            "level are not currently supported"
        )

    # Rescale datasets (only relevant for level>0)
    if not multiscales[0]["axes"][0]["name"] == "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f'metadata with axes={multiscales[0]["axes"]}. '
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=multiscales[0]["datasets"],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax for ax in multiscales[0]["axes"] if ax["type"] != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    image_group = zarr.group(zarr_url)
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(f"Helper function `prepare_label_group` returned {label_group=}")
    logger.info(f"Output label path: {zarr_url}/labels/{output_label_name}/0")
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")
    label_dtype = np.uint32

    # Ensure that all output shapes & chunks are 3D (for 2D data: (1, y, x))
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/398
    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {data_zyx.shape} " f"and chunks {data_zyx.chunks}"
    )

    # Initialize other things
    logger.info(f"Start ilastik pixel classification task for {zarr_url}")
    logger.info(f"relabeling: {relabeling}")
    logger.info(f"{data_zyx.shape}")
    logger.info(f"{data_zyx.chunks}")

    # Counters for relabeling
    num_labels_tot = {"num_labels_tot": 0}

    # Iterate over ROIs
    num_ROIs = len(list_indices)

    if output_ROI_table:
        bbox_dataframe_list = []

    logger.info(f"Now starting loop over {num_ROIs} ROIs")
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs}")

        # Prepare single-channel or dual-channel input for Ilastik
        if channel2.is_set():
            # Dual channel mode
            img_1 = load_region(
                data_zyx,
                region,
                compute=True,
                return_as_3D=True,
            )
            img_np = np.zeros((2, *img_1.shape))
            img_np[0, ...] = img_1
            img_np[1, ...] = load_region(
                data_zyx_c2,
                region,
                compute=True,
                return_as_3D=True,
            )
            logger.info(f"dual channel img shape {img_np.shape=}")
        else:
            img_np = load_region(data_zyx, region, compute=True, return_as_3D=True)
            img_np = img_np[
                np.newaxis, ...
            ]  # input for masked_wrapper has to be (czyx)
            logger.info(f"single channel img shape {img_np.shape=}")

        # Prepare keyword arguments for segment_ROI function
        kwargs_segment_ROI = {
            "shell": shell,
            "num_labels_tot": num_labels_tot,
            "foreground_class": foreground_class,
            "threshold": threshold,
            "min_size": min_size,
            "label_dtype": label_dtype,
            "relabeling": relabeling,
        }

        # Prepare keyword arguments for preprocessing function
        preprocessing_kwargs = {}
        if use_masks:
            preprocessing_kwargs = {
                "region": region,
                "current_label_path": f"{zarr_url}/labels/{output_label_name}/0",
                "ROI_table_path": ROI_table_path,
                "ROI_positional_index": i_ROI,
            }

        # Call segment_ROI through the masked-loading wrapper, which includes
        # pre/post-processing functions if needed
        new_label_img = masked_loading_wrapper(
            image_array=img_np,
            function=segment_ROI,
            kwargs=kwargs_segment_ROI,
            use_masks=use_masks,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        # Make 3D in case of 2D data
        # Check that the shape of the new label image matches the expected shape
        expected_shape = (
            e_z - s_z,
            e_y - s_y,
            e_x - s_x,
        )
        logger.info(f"Expected shape: {expected_shape}")
        if new_label_img.shape != expected_shape:
            try:
                new_label_img = np.broadcast_to(new_label_img, expected_shape)
            except Exception as err:
                raise ValueError(
                    f"Shape mismatch: {new_label_img.shape} != {expected_shape} "
                    "Between the segmented label image and expected shape in "
                    "the zarr array."
                ) from err

        if output_ROI_table:
            bbox_df = array_to_bounding_box_table(
                new_label_img,
                actual_res_pxl_sizes_zyx,
                origin_zyx=(s_z, s_y, s_x),
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = get_overlapping_pairs_3D(bbox_df, full_res_pxl_sizes_zyx)
            if len(overlap_list) > 0:
                logger.warning(
                    f"ROI {indices} has "
                    f"{len(overlap_list)} bounding-box pairs overlap"
                )

        # Compute and store 0-th level to disk
        da.array(new_label_img).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"End Ilastik pixel-classification task for {zarr_url}, "
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
        bbox_table = create_roi_table_from_df_list(bbox_dataframe_list)

        # Write to zarr group
        image_group = zarr.group(zarr_url)
        logger.info(
            "Now writing bounding-box ROI table to "
            f"{zarr_url}/tables/{output_ROI_table}"
        )
        table_attrs = {
            "type": "masking_roi_table",
            "region": {"path": f"../labels/{output_label_name}"},
            "instance_key": "label",
        }
        write_table(
            image_group,
            output_ROI_table,
            bbox_table,
            overwrite=overwrite,
            table_attrs=table_attrs,
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=ilastik_pixel_classification_segmentation,
        logger_name=logger.name,
    )
