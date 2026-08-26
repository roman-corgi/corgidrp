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

All branches should branch from the ``develop`` branch, and all pull requests should target the ``develop`` branch.


Writing a Pipeline Step
-----------------------

Each pipeline step in ``corgidrp`` is a function, typically contained in one of the ``lX_to_lY.py`` files (where X and Y are data levels). Think about how your feature can be implemented as a function that takes in some data and returns processed data.

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
* All additional function arguments and keywords must be one of: ``int``, ``float``, ``str``, or a class defined in ``corgidrp.data``.
    * (Long explanation for the curious: The reason for this is that pipeline steps can be written out as text files. Int/float/str are easily represented succinctly by textfiles. All classes in corgidrp.Data can be created simply by passing in a filepath. Therefore, all pipeline steps have easily recordable arguments for easy reproducibility.)
* The first line of the function generally should be creating a copy of the dataset (which will be the output dataset). This way, the output dataset is not the same instance as the input dataset. This will make it easier to ensure reproducibility.
* The function should always end by updating the header (and typically the data) of the output dataset, recording the processing step in the history.

See ``corgidrp.l2a_to_l2b.dark_subtraction`` as an example of a basic pipeline step.


Writing a Unit Test
-------------------

We are required to write tests to verify the functionality of the code. Instead of seeing this as an extra chore, I encourage you to write unit tests to be your debug script to get your code working (this is called "test-driven development").

All tests live in the ``tests/`` folder and each test is a function whose name starts with ``test_``. See ``tests/test_dark_sub.py`` as an example. Within each test you will typically:

1. Simulate mock data using the ``corgidrp.mocks`` module.
2. Run the data through the function you wrote.
3. Verify the output using ``assert`` statements.

Tests should cover the primary use cases of your code. Focus on ensuring the data is in the correct format, not on simulating data to high fidelity.

Importantly, these tests will allow code reviewers to test and understand your code. We will also run these tests in an automated test suite (continuous integration) with the pipeline to verify the functions continue to work (e.g., as dependencies change).


How to run your tests locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can either run indvidual tests yourself (to debug individual tests) or run the entire test suite to make sure you didn't break anything.

To run an individual test, call the test function you want to test at the bottom of its ``test_*.py`` script. Then, you just need to run the ``test_*.py`` script like any regular python file. See ``tests/test_dark_sub.py`` for an example.

To run all the unit tests in the test suite, go to the base corgidrp folder in a terminal and run the ``pytest`` command.

End-to-End Testing
------------------

End-to-end (e2e) testing processes data as one would with real telescope data, starting from L1. If applicable, write an e2e test following the examples in ``tests/e2e_tests/`` (e.g., ``l1_to_l2a_e2e.py``, ``l1_to_l2b_e2e.py``). For example, if you wrote a step that generates a calibration function, write an end-to-end test that produces the calibration file from L1 data. The steps are:

1. **Write a recipe** that produces the desired processed data product starting from L1 data. Determine the series of step functions and the arguments that should be modified. Refer to existing recipes in ``corgidrp/recipe_templates`` as examples and double check with the FDD.
2. **Obtain Simulated/TVAC L1 data**: For most cases there is no TVAC data, so you will need to use `corgisim <https://github.com/roman-corgi/corgisim>`_ to generate L1 data. See the `corgisim repo <https://github.com/roman-corgi/corgisim>`_ for more details. 
3. **Write an e2e test script** that processes the L1 data through your recipe using ``corgidrp.walker``:

   * You will likely need to modify ``corgidrp.walker.guess_template()`` to add logic for selecting your recipe from header keywords (e.g., ``VISTYPE``). Ask Jason if it is unclear what to do.
   * Your recipe may require calibration files; create them as part of the setup process in the script (see ``tests/e2e_tests/l1_to_l2b_e2e.py`` for examples). Note that the DRP now comes with a suite of default calibration files for L1 to L2b processing, so you probably dont' need to remake those. 
   * See the existing tests in ``tests/e2e_tests/`` for how to structure this e2e test script. You should only need to write a single script.

4. **Test the script locally** and debug as necessary. The chunk of code at the bottom of each e2e test (starting with ``if __name__ == "__main__"``) is used for debugging, allowing you to directly call the e2e test like a regular Python script. 
5. **Upload your L1 data to Sharepoint**: put your simulated L1 data into the "DRP E2E Test Data v2" folder. Ask Jason for access to the sharepoint. 
7. **Open a PR.**

To run the existing e2e tests locally, download the DRP E2E Test Data from sharepoint. You can also download the latest release from `Zenodo <https://zenodo.org/records/20634613>`_, although this will not have potential new L1 data added since the last release. After you have the necessary data downloaded, you can run all the necessary e2e tests by running:

.. code-block:: bash

    pytest --which e2e --e2edata_path /path/to/CGI_TVAC_Data --e2eoutput_path tests/e2e_tests/ tests/e2e_tests/


Linting
-------

In addition to unit tests, your code must pass a static analysis check before it can be merged. ``corgidrp`` runs a subset of ``flake8`` tests. Replicate locally by running the following command from the corgidrp root directory:

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

Before opening a pull request, review the :ref:`design-principles` below. Use the GitHub pull request feature to request that your changes be merged into ``develop``. Assign Jason or Max as reviewers. If you get chang erequests, you can push additional commits to your branch to update the PR (you don't need to delete the PR and make a new one).

Pre-PR checklist:

* If working from a fork, sync your fork to the upstream repository.
* Ensure your branch can be merged automatically by merging ``develop`` into your branch first.
* Ensure all new code has properly formatted docstrings (the ``flake8`` command above will catch violations).
* Ensure all commits have informative messages.
* Ensure all unit tests pass locally and that you have added new tests for all new functionality.
* Write a PR description that fully describes all changes and additions, and references the issue number (e.g., ``Closes #123``).


.. _design-principles:

Overarching Design Principles
------------------------------

* **Minimize external packages**, unless it saves a lot of development time. If you need one, default to well-established packages such as ``numpy``, ``scipy``, or ``astropy``. If you think you need to use something else, please check with Jason and Max.
* **Minimize new classes**, with the exception of new classes that extend the existing ``data`` framework.
* **Module size.** Each ``*.py`` file should hold roughly 5–15 functions. New modules should be general enough in topic to accommodate other future functions.
* **Data format standards**: All the image data in the Dataset and Image objects should be saved as standard numpy arrays. Other types of arrays (such as masked arrays) can be used as intermediate products within a function.
* **Keep things simple.** 
* **Use descriptive variable names always.** 
* **Documentation with comments**: Comments should be used to describe a section of the code where it's not immediately obvious what the code is doing. Using descriptive variable names will minimize the amount of comments required.
* **Make pull requests small.** Smaller PRs are easier to review. You can submit multiple PRs for the same task (e.g., a first simple implementation followed by a separate PR adding options). You can submit multiple PRs for multiple small fixes. 


FAQ
---

Does my pipeline function need to save files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. Files are saved by higher-level pipeline code. As long as your function returns an instance of a ``corgidrp.data`` class, it will have a ``save()`` method that the framework calls automatically.

Can I create new data classes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. New data classes should generally be subclasses of the ``Image`` class (see the ``Dark`` class as an example). Each calibration type should have its own ``Image`` subclass. Talk with Jason and Max about how your class should be implemented. You do not necessarily need to write a ``copy()`` method for ``Image`` subclasses; the base class ``copy()`` is sufficient for calibration objects.

What Python version should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.12.

How should I treat different kinds of function parameters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Constants** — things highly unlikely to change can be defined directly in modules such as ``detector.py``.
* **System properties** — measured values that may change over time belong in a calibration file such as ``DetectorParams``.
* **Function parameters** — choices that apply to a specific function where you want flexibility depending on use case (e.g., which detector area to use) should be keyword arguments to that function.
* **Pipeline behavior settings** — things like the location of the calibration database or whether to save individual error terms belong in the config file.

Where should I store computed variables so they can be referenced later?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If possible, store them in the header of the dataset being processed or in a new HDU extension. If neither works, discuss with Jason or Max.

Where do I save test data files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auxiliary data for tests should go in ``tests/test_data/``. Files larger than 1 MB should be stored using Git LFS (ask Jason about setup).
