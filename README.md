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

