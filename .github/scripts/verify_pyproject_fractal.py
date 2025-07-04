import tomllib
import sys


if __name__ == "__main__":

    # Load the main `pyproject.toml` file
    with open("pyproject.toml", "rb") as fp:
        manifest = tomllib.load(fp)

    # Load the fractal-specific `pyproject-fractal.toml` file
    with open("pyproject-fractal.toml", "rb") as fp:
        manifest_fractal = tomllib.load(fp)

    # Compare `pyproject-fractal.toml` with a sanitized version of
    # `pyproject.toml`
    manifest["tool"]["pixi"]["workspace"]["platforms"] = ["linux-64"]
    environments = manifest["tool"]["pixi"]["environments"]
    manifest["tool"]["pixi"]["environments"] = {
            key: value
            for key, value in environments.items()
            if key == "default"
        }
    if manifest == manifest_fractal:
        print("All good.")
    else:
        sys.exit("ERROR: please update `pyproject-manifest.toml`.")