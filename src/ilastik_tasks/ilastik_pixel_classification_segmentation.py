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
"""

import logging
from typing import Any, Optional

import anndata as ad
import dask.array as da
import fractal_tasks_core
import numpy as np
import skimage.measure
import vigra
import zarr
from fractal_tasks_core.channels import ChannelInputModel, get_channel_from_image_zarr
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
from pydantic import validate_call

logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def seutp_ilastik(model_path: str):
    """Setup Ilastik headless shell."""
    args = app.parse_args([])
    args.headless = True
    args.project = model_path
    args.readonly = True
    shell = app.main(args)
    return shell


def segment_ROI(
    input_data: np.ndarray,
    shell: Any,
    threshold: int = 10000,
    min_size: int = 3,
) -> np.ndarray:
    """Run the Ilastik model on a single ROI.

    Args:
        input_data: Input data. Shape (z, y, x).
        shell: Ilastik headless shell.
        threshold: Threshold for the Ilastik model.
        min_size: Minimum size for the Ilastik model.

    Returns:
        np.ndarray: Segmented image. Shape (z, y, x).
    """
    # run ilastik headless
    logger.info(f"{input_data.shape=}")

    # reformat as tzyxc data expected by ilastik

    input_data = input_data[np.newaxis, :, :, :, np.newaxis]
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
    logger.info(f"{ilastik_output.shape=}")

    # reformat to 2D
    ilastik_output = np.reshape(
        ilastik_output, (input_data.shape[1], input_data.shape[2], input_data.shape[3])
    )
    logger.info(f"{ilastik_output.shape=}")

    # take mask of regions above threshold
    ilastik_output[ilastik_output < threshold] = 0
    ilastik_output[ilastik_output >= threshold] = 1

    # label image
    ilastik_labels = skimage.measure.label(ilastik_output)

    # remove objects below min_size - also removes anything with major or minor axis
    # length of 0 for compatibility with current measurements task (01.24)
    if min_size > 0:
        label_props = skimage.measure.regionprops(ilastik_labels)
        labels2remove = [
            label_props[i].label
            for i in range(ilastik_labels.max())
            if (label_props[i].area < min_size)
            or (label_props[i].axis_major_length < 1)
            or (label_props[i].major_axis_length < 1)
        ]
        print(f"number of labels before filtering for size = {ilastik_labels.max()}")
        ilastik_labels[np.isin(ilastik_labels, labels2remove)] = 0
        ilastik_labels = skimage.measure.label(ilastik_labels)
        print(f"number of labels after filtering for size = {ilastik_labels.max()}")
        label_props = skimage.measure.regionprops(ilastik_labels)

    return ilastik_labels


@validate_call
def ilastik_pixel_classification_segmentation(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    level: int,
    channel: ChannelInputModel,
    ilastik_model: str,
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    # Cellpose-related arguments
    threshold: int = 10000,
    min_size: int = 3,
    use_masks: bool = True,
    overwrite: bool = True,
) -> None:
    """Run Ilastik Pixel Classification on a Zarr image.

    Args:
        zarr_url: URL of the Zarr image.
        level: Level of the Zarr image to process.
        channel: Channel input model.
        ilastik_model: Path to the Ilastik model.
        input_ROI_table: Name of the input ROI table.
        output_ROI_table: Name of the output ROI table.
        output_label_name: Name of the output label.
        threshold: Threshold for the Ilastik model.
        min_size: Minimum size for the Ilastik model.
        use_masks: Whether to use masks.
        overwrite: Whether to overwrite existing data.

    """
    logger.info(f"Processing {zarr_url=}")

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
    shell = seutp_ilastik(ilastik_model)

    # Find channel index
    omero_channel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        label=channel.label,
        wavelength_id=channel.wavelength_id,
    )
    if omero_channel:
        ind_channel = omero_channel.index
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
    # Workaround for #788: Only load channel index when there is a channel
    # dimension
    if ngff_image_meta.axes_names[0] != "c":
        data_zyx = da.from_zarr(f"{zarr_url}/{level}")

    else:
        data_zyx = da.from_zarr(f"{zarr_url}/{level}")[ind_channel]

    logger.info(f"{data_zyx.shape=}")

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

    # Rescale datasets (only relevant for level>0)
    # Workaround for #788
    if ngff_image_meta.axes_names[0] != "c":
        new_datasets = rescale_datasets(
            datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
            coarsening_xy=coarsening_xy,
            reference_level=level,
            remove_channel_axis=False,
        )
    else:
        new_datasets = rescale_datasets(
            datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
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
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
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

        # Prepare single-channel or dual-channel input for cellpose
        img_np = load_region(data_zyx, region, compute=True, return_as_3D=True)

        # Prepare keyword arguments for segment_ROI function
        kwargs_segment_ROI = {
            "shell": shell,
            "threshold": threshold,
            "min_size": min_size,
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
        f"End cellpose_segmentation task for {zarr_url}, " "now building pyramids."
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
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=ilastik_pixel_classification_segmentation,
        logger_name=logger.name,
    )
