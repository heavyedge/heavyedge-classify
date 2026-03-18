.. HeavyEdge-Classify documentation master file, created by
   sphinx-quickstart on Tue Mar 17 20:23:57 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

********************************
HeavyEdge-Classify documentation
********************************

HeavyEdge-Classify is a Python package for probabilistic classification of coating edge profiles.

.. note::

   This package provides only the model architecture and command line interfaces.
   It does not include any pre-trained model or training data.

Usage
=====

HeavyEdge-Classify is designed to be used either as a command line program or as a Python module.

Command line
------------

Command line interface provides pre-defined subroutines for training and prediction.
It can be invoked by:

.. code-block:: bash

   heavyedge classify-train <args>
   heavyedge classify-predict <args>

Refer to help message of each command for their arguments.

Python module
-------------

The Python module :mod:`heavyedge_classify` provides functions and classes for Python runtime.
Refer to :ref:`api` section for high-level interface.

Module reference
================

.. module:: heavyedge_classify

This section provides reference for :mod:`heavyedge_classify` Python module.

.. _api:

Runtime API
-----------

.. automodule:: heavyedge_classify.api
    :members:

Low-level API
-------------

.. automodule:: heavyedge_classify.model
    :members:
