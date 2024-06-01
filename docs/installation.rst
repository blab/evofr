Installation Guide
==================

This page provides detailed instructions on how to install `evofr`, a Python package for evolutionary forecasting of genetic variants.

Installing with pip
-------------------

The easiest way to install the latest version of `evofr` is via pip. This method will automatically handle all dependencies, including JAX and Numpyro:

.. code-block:: bash

    pip install evofr

Installing from Source
----------------------

For those who prefer to install `evofr` from the source or want to contribute to the development, follow these steps:

.. code-block:: bash

    git clone https://github.com/blab/evofr.git
    cd evofr
    pip install .

Installing from Source Using Poetry
-----------------------------------

Poetry is a tool for dependency management and packaging in Python. To use Poetry to install `evofr` from the source, follow these instructions:

.. code-block:: bash

    git clone https://github.com/blab/evofr.git
    cd evofr
    poetry install

This command will create a virtual environment and install all dependencies defined in `pyproject.toml`. To activate the virtual environment created by Poetry, you can use:

.. code-block:: bash

    poetry shell

Alternatively, to run commands within the virtual environment without activating it, use `poetry run`:

.. code-block:: bash

    poetry run python -m mymodule

For more information on using Poetry, visit the `Poetry documentation <https://python-poetry.org/docs/>`_.

Environment Setup
-----------------

It is often beneficial to set up a virtual environment for Python projects to manage dependencies separately from the system-wide installations:

.. code-block:: bash

    python -m venv evofr-env
    source evofr-env/bin/activate
    pip install evofr

This method is especially recommended when working on development or managing multiple Python packages.

Additional Installation Help
----------------------------

If you encounter any issues during the installation, particularly related to JAX, please consult the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for detailed instructions and troubleshooting tips.