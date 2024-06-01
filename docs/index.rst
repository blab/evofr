.. evofr documentation master file, created by
   sphinx-quickstart on [date you ran the command].
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to evofr's documentation!
=================================

.. image:: https://github.com/blab/evofr/actions/workflows/ci.yaml/badge.svg
    :target: https://github.com/blab/evofr/actions/workflows/ci.yaml

Overview
--------

`evofr` is a Python package designed for evolutionary forecasting of genetic variants. 
This project provides tools aimed at modeling and predicting the evolution and prevalence of genetic variants over time. 
The package integrates various data handling, modeling, and plotting capabilities, making it a useful toolkit for researchers in the field of genetic epidemiology and evolutionary dynamics.

Installation
------------

You can install the latest release of `evofr` directly using pip:

.. code-block:: bash

    pip install evofr

For those looking to contribute or modify the package, it can be built locally with:

.. code-block:: bash

    poetry build
    pip install path-to-wheel>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   api_reference
   mlr_quickstart

=================================
