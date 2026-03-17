"""Manage sample data."""

import os
import subprocess
from importlib.resources import files

__all__ = [
    "get_sample_path",
    "make_all_samples",
    "clean_all_samples",
]


def get_sample_path(*paths, build=True):
    """Return path to sample data distributed by the package.

    If file does not exist, attempts to generate it using pre-defined recipe.

    Parameters
    ----------
    paths : list of str
        Subpath components.
    build : bool, default=True
        Attempt to build if file is missing.

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> from heavyedge_classify.samples import get_sample_path
    >>> print(get_sample_path())  #doctest: +SKIP
    heavyedge_classify/samples
    >>> print(get_sample_path("RawData"))  #doctest: +SKIP
    heavyedge_classify/samples/RawData
    """
    datapath = files("heavyedge_classify.samples")
    path = datapath.joinpath(*paths)

    if not os.path.exists(path) and build:
        from .recipes import RECIPES

        recipe = RECIPES.get("/".join(paths))
        if recipe is not None:
            commands = recipe(path)
            subprocess.run(commands, check=True)
    return path


def make_all_samples(progress=False):
    """Generate all sample data.

    Parameters
    ----------
    progress : bool, default=False
        Prints progress bar. Requires :mod:`tqdm`.
    """
    from .recipes import RECIPES

    if progress:
        from tqdm import tqdm

        items = tqdm(RECIPES.items())
    else:
        items = RECIPES.items()

    for file, recipe in items:
        path = get_sample_path(file, build=False)
        if not os.path.exists(path):
            commands = recipe(path)
            if commands is not None:
                subprocess.run(commands, check=True)


def clean_all_samples():
    """Delete all buildable samples."""
    from .recipes import RECIPES

    for file in RECIPES.keys():
        path = get_sample_path(file, build=False)
        path.unlink(missing_ok=True)
