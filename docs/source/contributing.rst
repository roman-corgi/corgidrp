Contributing
============

We encourage you to chat with Jason, Max, and Marie (e.g., on Slack) to discuss what to do before you get started. Brainstorming about how to implement something is a very good use of time and makes sure you aren't going down the wrong path. Contact Jason if you have any questions on how to get started on programming details (e.g., git).


Getting Started
---------------

Find a task to work on
~~~~~~~~~~~~~~~~~~~~~~~

Check out the `GitHub issues page <https://github.com/roman-corgi/corgidrp/issues>`_ for tasks that need attention. Alternatively, contact Jason (@semaphoreP). *Make sure to tag yourself on the issue and mention in the comments if you start working on it.*

Clone the repository and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the :doc:`installation` guide. Contact Jason (@semaphoreP) if you need write access to push changes. If you do not have write access you can contribute by `forking the repository <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ under your own GitHub user; replace ``roman-corgi`` with your username in the commands below. If you fork the repository, keep your fork in sync with the main repository (`syncing a fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork>`_).

To clone and install:

.. code-block:: bash

    git clone https://github.com/roman-corgi/corgidrp.git
    cd corgidrp
    pip install -e .

Create a feature branch
~~~~~~~~~~~~~~~~~~~~~~~~

Create a feature branch so you can develop without impacting other people's code. For example, if you're working on dark subtraction:

.. code-block:: bash

    git branch dark-sub
    git checkout dark-sub

All pull requests should target the ``develop`` branch.


Writing a Pipeline Step
-----------------------

Each pipeline step in ``corgidrp`` is a **pure function** contained in one of the ``lX_to_lY.py`` files (where X and Y are data levels). Think about how your feature can be implemented as a function that takes in some data and returns processed data.

All step functions should follow this template:

.. code-block:: python

    def example_step(dataset, calib_data, tuneable_arg=1, another_arg="test"):
        """
        Function docstrings are required and should follow Google style docstrings.
        We will not demonstrate it here for brevity.
        """
        # unless you don't alter the input dataset at all, plan to make a copy of the data
        # this is to ensure functions are reproducible
        processed_dataset = dataset.copy()

        ### Your code here that does the real work
        # here is a convenience field to grab all the data in a dataset
        all_data = processed_dataset.all_data
        ### End of your code that does the real work

        # update the header of the new dataset with your processing step
        history_msg = "I did an example step"
        # update the output dataset with the new data and update the history
        processed_dataset.update_after_processing_step(history_msg, new_all_data=all_data)

        # return the processed data
        return processed_dataset

The function body can do nearly anything, but the signature and structure must follow these rules:

* Each function must include a docstring describing what it does, its inputs (with units where appropriate), and its outputs (with units). Use `Google style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
* The input dataset must always be the first argument.
* Additional arguments and keywords are optional — many relevant parameters may already be in the ``Dataset`` headers. A step can have just the dataset as its only argument.
* All additional function arguments and keywords must be one of: ``int``, ``float``, ``str``, or a class defined in ``corgidrp.data``. (These types can be represented in text recipe files, making pipeline runs fully reproducible.)
* The first line of the function should typically create a copy of the input dataset to ensure reproducibility.
* The function should always end by updating the header (and typically the data) of the output dataset, recording the processing step in the history.

See ``corgidrp.l2a_to_l2b.dark_subtraction`` as an example of a basic pipeline step.


Writing a Unit Test
-------------------

Tests are required to verify the functionality of every pipeline step. Rather than viewing tests as an extra chore, write them as your debug script to get your code working — this is called *test-driven development*.

All tests live in the ``tests/`` folder and each test is a function whose name starts with ``test_``. See ``tests/test_dark_sub.py`` as an example. Within each test you will typically:

1. Simulate mock data using the ``corgidrp.mocks`` module.
2. Run the data through the function you wrote.
3. Verify the output using ``assert`` statements.

Tests should cover the primary use cases of your code. Focus on ensuring the data is in the correct format, not on simulating data to high fidelity.

Running tests locally
~~~~~~~~~~~~~~~~~~~~~

To run an individual test, call it at the bottom of its ``test_*.py`` script and run the script directly. To run the full test suite, go to the repository root and run:

.. code-block:: bash

    pytest tests/


End-to-End Testing
------------------

End-to-end (e2e) testing processes data as one would with real telescope data, starting from L1. If applicable, write an e2e test following the examples in ``tests/e2e_tests/`` (e.g., ``l1_to_l2a_e2e.py``, ``l1_to_l2b_e2e.py``). The steps are:

1. **Write a recipe** that produces the desired processed data product starting from L1 data. Determine the series of step functions and the arguments that should be modified. Refer to existing recipes in ``corgidrp/recipe_templates`` and the FDD.
2. **Obtain TVAC L1 data** from the Box folder (ask Alex Greenbaum or Jason if you don't have access). If no appropriate TVAC data exists (e.g., for boresight), write code that adds TVAC images to mock data to create realistic mock L1 data. Do not overwrite original data on Box.
3. **Write an e2e test script** that processes the L1 data through your recipe using ``corgidrp.walker``:

   * You will likely need to modify ``corgidrp.walker.guess_template()`` to add logic for selecting your recipe from header keywords (e.g., ``VISTYPE``). Ask Jason if it is unclear what to do.
   * Your recipe may require calibration files; create them as part of the setup process in the script (see ``tests/e2e_tests/l1_to_l2b_e2e.py`` for examples).
   * If you need mock L1 data, generate it in the script as well.

4. **Test the script locally** and debug as necessary. Where appropriate, compare results against the II&T/TVAC pipeline using the same input data.
5. **Measure resource usage.** Linux users can run ``/usr/bin/time -v python your_e2e_test.py``; Mac users can run ``/usr/bin/time -l -h -p python your_e2e_test.py``. Record elapsed (wall clock) time, CPU percentage (if parallelization was used), and peak memory (``Maximum resident set size``).
6. **Document your recipe** in the "Corgi-DRP Implementation Document" on Confluence (Section 2.0 table). Note any significant run time (> 1 minute) or memory usage (> 1 GB).
7. **Open a PR.**

To run the existing e2e tests locally, download the TVAC data and run:

.. code-block:: bash

    pytest --which e2e --e2edata_path /path/to/CGI_TVAC_Data --e2eoutput_path tests/e2e_tests/ tests/e2e_tests/


Linting
-------

In addition to unit tests, your code must pass a static analysis check before it can be merged. ``corgidrp`` runs a subset of ``flake8`` tests. Replicate locally from the repository root:

.. code-block:: bash

    flake8 . --count \
      --select=E9,F63,F7,F82,DCO020,DCO021,DCO022,DCO023,DCO024,DCO030,DCO031,DCO032,DCO060,DCO061,DCO062,DCO063,DCO064,DCO065 \
      --show-source --statistics

You need ``flake8`` and ``flake8-docstrings-complete`` installed (both are pip-installable). The set of checks may change over time; see the current configuration in ``.github/workflows/python-app.yml``.


AI Policy
---------
We require that you understand all code you contribute, regardless of how it is generated. We have the following rules with regard to code generated by AI tools: 

* Please review line-by-line all changes made by an AI coding tool. 
* Please make sure you understand line-by-line what all code generated by an AI coding tool is doing. Don't commit code you don't understand!
* If AI-generated code is unclear, refactor it to make the code easier to understand.

Please ease the burden on the code maintainers by reviewing and cleaning up all AI-generated code before you submit a PR!


Creating a Pull Request
-----------------------

Before opening a pull request, review the :ref:`design-principles` below. Use the GitHub pull request feature to request that your changes be merged into ``develop``. Assign Jason or Max as reviewers. You can push additional commits to update the PR without closing and reopening it.

Pre-PR checklist:

* If working from a fork, sync your fork to the upstream repository.
* Ensure your branch can be merged automatically (merge ``develop`` into your branch first if needed).
* Ensure all new code has properly formatted docstrings (the ``flake8`` command above will catch violations).
* Ensure all commits have informative messages.
* Ensure all unit tests pass locally and that you have added new tests for all new functionality.
* Write a PR description that fully describes all changes and additions, and references the issue number (e.g., ``Fixes #123``).


.. _design-principles:

Overarching Design Principles
------------------------------

* **Minimize external packages.** If you need one, default to well-established packages such as ``numpy``, ``scipy``, or ``astropy``. Anything else requires sign-off from Jason and Max.
* **Minimize new classes.** Exceptions are new classes that extend the existing data framework.
* **Module size.** Each ``*.py`` file should hold roughly 5–10 functions. New files should be general enough in topic to accommodate other future functions.
* **Use standard numpy arrays** for all image data in ``Dataset`` and ``Image`` objects. Masked arrays and other types may be used as intermediate products within a function.
* **Keep things simple.** Use descriptive variable names always. Use comments only to explain sections where the intent is not immediately obvious from the code.
* **Make pull requests small.** Smaller PRs are easier to review. You can submit multiple PRs for the same task (e.g., a first simple implementation followed by a separate PR adding options).


FAQ
---

Does my pipeline function need to save files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. Files are saved by higher-level pipeline code. As long as your function returns an instance of a ``corgidrp.data`` class, it will have a ``save()`` method that the framework calls automatically.

Can I create new data classes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. New data classes should generally be subclasses of the ``Image`` class — see the ``Dark`` class as an example. Each calibration type should have its own ``Image`` subclass. Talk with Jason and Max about how your class should be implemented. You do not necessarily need to write a ``copy()`` method for ``Image`` subclasses; the base class ``copy()`` is sufficient for calibration objects.

What Python version should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.12.

How should I treat different kinds of function parameters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Constants** — things highly unlikely to change — can be defined directly in modules such as ``detector.py``.
* **System properties** — measured values that may change over time — belong in a calibration file such as ``DetectorParams``.
* **Function parameters** — choices that apply to a specific function and where you want flexibility (e.g., which detector area to use) — should be keyword arguments to that function.
* **Pipeline behavior settings** — things like the location of the calibration database or whether to save individual error terms — belong in the config file.

Where should I store computed variables so they can be referenced later?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If possible, store them in the header of the dataset being processed or in a new HDU extension. If neither works, discuss with Jason.

Where do I save test data files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auxiliary data for tests should go in ``tests/test_data/``. Files larger than 1 MB should be stored using Git LFS (ask Jason about setup).
