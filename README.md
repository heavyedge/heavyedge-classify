# HeavyEdge-Classify

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/heavyedge-classify.svg)](https://pypi.python.org/pypi/heavyedge-classify/)
[![PyPI Version](https://img.shields.io/pypi/v/heavyedge-classify.svg)](https://pypi.python.org/pypi/heavyedge-classify/)
[![License](https://img.shields.io/github/license/heavyedge/heavyedge-classify)](https://github.com/heavyedge/heavyedge-classify/blob/master/LICENSE)
[![CI](https://github.com/heavyedge/heavyedge-classify/actions/workflows/ci.yml/badge.svg)](https://github.com/heavyedge/heavyedge-classify/actions/workflows/ci.yml)
[![CD](https://github.com/heavyedge/heavyedge-classify/actions/workflows/cd.yml/badge.svg)](https://github.com/heavyedge/heavyedge-classify/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/heavyedge-classify/badge/?version=latest)](https://heavyedge-classify.readthedocs.io/en/latest/?badge=latest)

Python package for probabilistic classification of edge profiles.

## Usage

HeavyEdge-Classify provides simple command line interfaces:

```bash
heavyedge classify-train profiles labels
heavyedge classify-predict profiles model
```

Python Runtime API is also provided to incorporate the trained model into other framework.
Refer to the package documentation for more information.

## Installation

```
$ pip install heavyedge-classify
```

## Documentation

The manual can be found online:

> https://heavyedge-classify.readthedocs.io

If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Developing

### Installation

For development features, you must install the package by `pip install -e .[dev]`.

### Testing

Run `pytest` command to perform unit test.

When doctest is run, buildable sample data are rebuilt by default.
To disable this, set `HEAVYEDGE_TEST_REBUILD` environment variable to zero.
For example,
```
HEAVYEDGE_TEST_REBUILD=0 pytest
```
