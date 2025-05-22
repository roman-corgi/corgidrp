Installation Guide
==================

Quick Installation
-------------------

As the code is very much still in development, clone this repository, enter the top-level folder, and run the following command:

.. code-block:: bash

    pip install -e .

Then you can import ``corgidrp`` like any other Python package!

The installation will create a configuration folder in your home directory called ``.corgidrp``. 
That configuration directory will be used to locate things on your computer such as the location of the calibration database and the pipeline configuration file. The configuration file stores settings such as whether to track each individual error term added to the noise.

For Developers
---------------
Large binary files (used in tests) are stored in Git LFS. Install Git LFS if it isn't already installed. 

`Install Git LFS <https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>`_ if it isn't already installed. 

You may need to run the following after checking out the repository to download the latest large binary files, or the unit tests may fail:

.. code-block:: bash

    git lfs pull

To run the existing end-to-end tests, you also need the II&T code, which is used directly for comparing results. This also requires Git LFS to be installed first. Then install the II&T code by doing the following while in the top-level folder:

.. code-block:: bash

    pip install -r requirements_e2etests.txt corgidrp

This will install the II&T repositories ``cal`` and ``proc_cgi_frame``.

Troubleshooting
----------------
If you run into any issues with things in the ``.corgidrp`` directory not being found properly when you run the pipeline, such as a ``DetectorParams`` file, ``caldb``, or configuration settings, your ``corgidrp`` is configured into a weird state. Report the bug to our GitHub issue tracker, including both the error message and the state of your ``.corgidrp`` folder. 

If you don't want to wait for us to troubleshoot the bug and deploy a fix, you can probably resolve the issue by completely deleting your ``.corgidrp`` folder and rerunning the code (the code will automatically remake it). This, however, means you will lose any changes you've made to your settings as well as your calibration database.

