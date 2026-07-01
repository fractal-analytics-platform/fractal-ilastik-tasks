# ilastik-tasks

Collection of Fractal task to run Headless ilastik workflows.

## Tasks

1. **Pixel Classification Segmentation**: A task to run a pixel classification workflow in headless mode. The task requires a trained ilastik project file and a list of input images.
    The task will run the pixel classification workflow on the input images, and label the connected components in the output image.

## Installation and Deployment

* Install `pixi` package manager [instructions](https://pixi.sh/latest/installation/)
* run the task:

```bash
pixi run python some_script.py
```

if you need to use the `dev` version of the package, you can run:

```bash
pixi run -e dev python some_script.py
```

### Deploying on a Fractal server

`ilastik-tasks` depends on `ilastik-core` and `vigra`, which are conda-only
packages and are **not published on PyPI**. Because of this, the task must be
collected on the Fractal server using the **Pixi** task-collection method,
rather than the `pip`/local-whl method (the latter cannot resolve these
dependencies and will fail at runtime, e.g. with `ModuleNotFoundError: No
module named 'numpy'`).

To install it, download the `.tar.gz` source archive (not the `.whl`) of the
desired version from the [Releases page](https://github.com/fractal-analytics-platform/fractal-ilastik-tasks/releases),
and use it as the input for the Pixi task collection on the Fractal server.
