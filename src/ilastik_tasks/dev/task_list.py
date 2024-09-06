"""Contains the list of tasks available to fractal."""

from fractal_tasks_core.dev.task_models import ParallelTask

TASK_LIST = [
    ParallelTask(
        name="Thresholding Label Task",
        executable="thresholding_label_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
]
