"""This is the Python module for my_task."""

import logging
from typing import Any, Optional

import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr
from fractal_tasks_core.channels import (
    ChannelInputModel,
    OmeroChannel,
    get_channel_from_image_zarr,
)
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.utils import rescale_datasets
from pydantic import validate_call
from skimage.measure import label
from skimage.morphology import ball, dilation, opening, remove_small_objects

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


@validate_call
def thresholding_label_task(
    *,
    zarr_url: str,
    threshold: int,
    channel: ChannelInputModel,
    label_name: Optional[str] = None,
    min_size: int = 50,
    overwrite: bool = True,
) -> None:
    """Threshold an image and find connected components.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        threshold: Threshold value to be applied.
        channel: Channel to be thresholded.
        label_name: Name of the resulting label image
        min_size: Minimum size of objects. Smaller objects are filtered out.
        overwrite: Whether to overwrite an existing label image
    """
    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

    # Parse and log several NGFF-image metadata attributes
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    logging.info(f"  Axes: {ngff_image_meta.axes_names}")
    logging.info(f"  Number of pyramid levels: {ngff_image_meta.num_levels}")
    logging.info(
        "Full-resolution ZYX pixel sizes (micrometer): "
        f"{ngff_image_meta.get_pixel_sizes_zyx(level=0)}"
    )

    # Find the channel metadata
    channel_model: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=channel.wavelength_id,
        label=channel.label,
    )

    # Set label name
    if not label_name:
        label_name = f"{channel_model.label}_thresholded"

    # Load the highest-resolution multiscale array through dask.array
    array_zyx = da.from_zarr(f"{zarr_url}/0")[channel_model.index]
    logging.info(f"{array_zyx=}")

    # Process the image with an image processing approach of your choice
    label_img = process_img(
        array_zyx.compute(),
        threshold=threshold,
        min_size=min_size,
    )

    # Prepare label OME-Zarr
    # If the resulting label image is of lower resolution than the intensity
    # image, set the downsample variable to the number of downsamplings
    # required (e.g. 2 if the image is downsampled 4x per axis with an
    # ngff_image_meta.coarsening_xy of 2)
    label_attrs = generate_label_attrs(ngff_image_meta, label_name, downsample=0)
    label_group = prepare_label_group(
        image_group=zarr.group(zarr_url),
        label_name=label_name,
        label_attrs=label_attrs,
        overwrite=overwrite,
    )
    # Write the processed array back to the same full-resolution Zarr array
    label_group.create_dataset(
        "0",
        data=label_img,
        overwrite=overwrite,
        dimension_separator="/",
        chunks=array_zyx.chunksize,
    )

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{label_name}",
        overwrite=True,
        num_levels=ngff_image_meta.num_levels,
        coarsening_xy=ngff_image_meta.coarsening_xy,
        aggregation_function=np.max,
    )


def process_img(int_img: np.array, threshold: int, min_size: int = 50) -> np.array:
    """Image processing function, to be replaced with your custom logic

    Numpy image & parameters in, label image out

    Args:
        int_img: Intensity image as a numpy array
        threshold: Thresholding value to binarize the image
        min_size: Object size threshold for filtering

    Returns:
        label_img: np.array
    """
    # Thresholding the image
    binary_img = int_img >= threshold

    # Removing small objects
    cleaned_img = remove_small_objects(binary_img, min_size=min_size)
    # Opening to separate touching objects
    selem = ball(1)
    opened_img = opening(cleaned_img, selem)

    # Optional: Dilation to restore object size
    dilated_img = dilation(opened_img, selem)

    # Labeling the processed image
    label_img = label(dilated_img, connectivity=1)

    return label_img


def generate_label_attrs(
    ngff_image_meta: NgffImageMeta, label_name: str, downsample: int = 0
) -> dict[str, Any]:
    """Generates the label OME-zarr attrs based on the image metadata

    Args:
        ngff_image_meta: image meta object for the corresponding NGFF image
        label_name: name of the newly generated label
        downsample: How many levels the label image is downsampled from the
            ngff_image_meta image (0 for no downsampling, 1 for downsampling
            once by the coarsening factor etc.)

    Returns:
        label_attrs: Dict of new OME-Zarr label attrs

    """
    new_datasets = rescale_datasets(
        datasets=[
            dataset.dict(exclude_none=True) for dataset in ngff_image_meta.datasets
        ],
        coarsening_xy=ngff_image_meta.coarsening_xy,
        reference_level=downsample,
        remove_channel_axis=True,
    )
    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": label_name,
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
    return label_attrs


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=thresholding_label_task)
