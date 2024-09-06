"""Contains the list of tasks available to fractal."""

from fractal_tasks_core.dev.task_models import ParallelTask

TASK_LIST = [
    ParallelTask(
        name="Ilastik Pixel Classification Segmentation",
        executable="ilastik_pixel_classification_segmentation.py",
        meta={"cpus_per_task": 8, "mem": 8000},
    ),
]
