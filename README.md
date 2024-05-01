# corgidrp: CORonaGraph Instrument Data Reduction Pipeline
This is the data reduction pipeline for the Nancy Grace Roman Space Telescope Coronagraph Instrument

![Testing Badge](https://github.com/roman-corgi/corgidrp/actions/workflows/python-app.yml/badge.svg)

## Install
As the code is very much still in development, clone this repository, enter the top-level folder, and run the following command:
```
pip install -e .
```
Then you can import `corgidrp` like any other python package!

The installation will create a configuration folder in your home directory called `.corgidrp`. 
That configuration directory will be used to locate things on your computer such as the location of the calibration database and the pipeline configuration file. The configuration files stores setting such as whether to track each individual error term added to the noise. 

## How to Contribute

We encourage you to chat with Jason, Max, and Marie (e.g., on Slack) to discuss what to do before you get started. Brainstorming
about how to implement something is a very good use of time and makes sure you aren't going down the wrong path. 
Contact Jason is you have any questions on how to get started on programming details (e.g., git). 

Below is a quick tutorial  that outlines the general contribution process. 

### The basics of getting setup
#### Find a task to work on
Check out the [Github issues page](https://github.com/roman-corgi/corgidrp/issues) for tasks that need attention. Alternatively, contact Jason (@semaphoreP). _Make sure to tag yourself on the issue and mention in the comments if you start working on it._ 

#### Clone the git repository and install

See install instructions above. Contact Jason (@semaphoreP) if you need write access to push changes as you make edits.  If you do not have write access to the repository, you can still contribute by creating a fork of the repository under your own GitHub user.  See here for details: https://docs.github.com/en/get-started/quickstart/fork-a-repo.  You can then use the same commands given below, but just replace `roman-corgi` with your own GitHub username. If you fork the repository, you will need to make sure that your fork is up to date with the main repository (https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

To quickly get up an running with the repository, execute the following commands in a terminal (or command prompt - you will need to have the git executable installed on your system):
```
> git clone https://github.com/roman-corgi/corgidrp.git
> cd corgidrp
> pip install -e .
```

##### Make a new git branch to work on

You will create a "feature branch" so you can develop your feature without impacting other people's code. Let's say I'm working on dark subtraction. I could create a feature branch and switch to it like this

```
> git branch dark-sub
> git checkout dark-sub
```

### Write your pipeline step

In `corgidrp`, each pipeline step is a function, that is contained in one of the lX_to_lY.py files (where X and Y are various data levels). 
Think about how your feature can be implemented as a function that takes in some data and outputs processed data. Please see below for some 
`corgidrp` design principles. 

All functions should follow this example:

```
def example_step(dataset, calib_data, tuneable_arg=1, another_arg="test"):
    """
    Function docstrings are required and should follow Google style docstrings. 
    We will not demonstrate it here for brevity.
    """
    # unless you don't alter the input dataset at all, plan to make a copy of the data
    # this is to ensure functions are reproducible
    processed_dataset = input_dataset.copy()

    ### Your code here that does the real work
    # here is a convience field to grab all the data in a dataset
    all_data = processed_dataset.all_data
    ### End of your code that does the real work

    # update the header of the new dataset with your processing step
    history_msg = "I did an example step"
    # update the output dataset with the new data and update the history
    processed_dataset.update_after_processing_step(history_msg, new_all_data=all_data)

    # return the processed data
    return processed_dataset
```

Inside the function can be nearly anything you want, but the function signature and start/end of the function should follow a few rules.

  * Each function should include a docstring that descibes what the function is doing, what the inputs (including units if appropriate) are and what the outputs (also with units). The dosctrings should be [goggle style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). 
  * The input dataset should always be the first input
  * Additional arguments and keywords exist only if you need them--many relevant parameters might already by in Dataset headers. A pipeline step can only have a single argument (the input dataset) if needed.
  * All additional function arguments/keywords should only consist of the following types: int, float, str, or a class defined in corgidrp.Data. 
    * (Long explaination for the curious: The reason for this is that pipeline steps can be written out as text files. Int/float/str are easily represented succinctly by textfiles. All classes in corgidrp.Data can be created simply by passing in a filepath. Therefore, all pipeline steps have easily recordable arguments for easy reproducibility.)
  * The first line of the function generally should be creating a copy of the dataset (which will be the output dataset). This way, the output dataset is not the same instance as the input dataset. This will make it easier to ensure reproducibility. 
  * The function should always end with updating the header and (typically) the data of the output dataset. The history of running this pipeline step should be recorded in the header. 

You can check out `corgidrp.l2a_to_l2b.dark_subtraction` function as an example of a basic pipeline step.

### Write a unit test to debug your pipeline step

We are required to write tests to verify the functionality of the code. Instead of seeing this as an extra chore, I encourage you to write unit tests to be your debug script to get your code working (this is called "test-driven development"). 

All tests are stored in the `tests` folder and each test is a function that starts with `test_`. See `tests/test_dark_sub.py` as an example. Within each test, you will likely need to simulate some mock data, run it through your function you wrote, and verify it ran correctly using assert statements. Your tests should cover the primary use cases of your code, and check that the function outputs what you expect. You do not need high fidelity data for your test: focus on making sure the data is in the correct format as real data, and less on making sure the data values are simulated to high fidelity (see the examples in the `mocks.py` module). 

Importantly, these tests will allow code reviewers to test and understand your code. We will also run these tests in an automated test suite (continuous integration) with the pipeline to verify the functions continue to work (e.g., as dependencies change).

#### How to run your tests locally

You can either run tests individually yourself (to debug individual tests) or run the entire test suite to make sure you didn't break anything.

To run an individual test, call the test function you want to test at the bottom of its `test_*.py` script. Then, you just need to run the `test_*.py` script. See `tests/test_dark_sub.py` for an example.

To run all the tests in the test suite, go to the base corgidrp folder in a terminal and run the `pytest` command. 

### Linting

In addition to unit tests, your code will need to pass a static analysis before being merged.  `corgidrp` currently runs a subset of flake8 tests, which you can replicate on your local system by running:

```
flake8 . --count --select=E9,F63,F7,F82,DCO020,DCO021,DCO022,DCO023,DCO024,DCO030,DCO031,DCO032,DCO060,DCO061,DCO062,DCO063,DCO064,DCO065 --show-source --statistics
```
from the top-level directory of the repository.  In order to run these tests you will need to have `flake8` and `flake8-docstrings-complete` installed (both are pip-installable).  Note that the test subset may be updated in the future.  To see the current set of tests being applied, look in the continuous integration GitHub action, located in the repository in file `.github/workflows/python-app.yml`.

### Create a pull request to merge your changes
Before creating a pull request, review the design Principles below. Use the Github pull request feature to request that your changes get merged into the `main` branch. Assign Jason/Max to be your reviewers. Your changes will be reviewed, and possibly some edits will be requested. You can simply make additional pushes to your branch to update the pull request with those changes (you don't need to delete the PR and make a new one). When the branch is satisfactory, we will pull your changes in. When preparing your pull request, you may find it helpful to follow this checklist:

- [ ] If working with a fork, sync your fork to the upstream repository
- [ ] Ensure that your pull can be merged automatically by merging main onto the branch you wish to pull from
- [ ] Ensure that all of your new additions have properly formatted docstrings (this will happen automatically if you run the `flake8` command given above
- [ ] Ensure that all of the commits going in to your pull have informative messages
- [ ] Ensure that all unit tests pass locally, and that you have provided new unit tests for all new functionality in your pull
- [ ] Create a new pull request, fully describing all changes/additions


## Overarching Design Principles
* Minimize the use of external packages, unless it saves us a lot of time. If you need to use something external, default to well-established and maintained packages, such as `numpy`, `scipy` or `Astropy`. If you think you need to use something else, please check with Jason and Max. 
* Minimize the use of new classes, with the exception of new classes that extend the existing data framework.
* The python module files (i.e. the *.py files) should typically hold on the order of 5-10 different functions. You can create new ones if need be, but the new files should be general enough in topic to encapsulate other future functions as well.
* All the image data in the Dataset and Image objects should be saved as standard numpy arrays. Other types of arrays (such as masked arrays) can be used as intermediate products within a function.
* Keep things simple
* Use _descriptive_ variable names **always**.
* Comments should be used to describe a section of the code where it's not immediately obvious what the code is doing. Using descriptive variable names will minimize the amount of comments required.

## FAQ

  * Does my pipeline function need to save files?
    * Files will be saved by a higher level pipeline code. As long as you output an object that's an instance of a `corgidrp.Data` class, it will have a `save()` function that will be used.
  * Can I create new data classes?
    * Yes, you can feel free to make new data classes. Generally, they should be a subclass of the `Image` class, and you can look at the `Dark` class as an example. Each calibration type should have its own `Image` subclass defined. Talk with Jason and Max to discuss how your class should be implemented!

* What python version should I develop in?
  * Python 3.12
    
* How should I treat hard-coded variables
  * If a variable value is extremely unlikely to change AND is only required by one module, it can be hard coded inside that module.
  * If it is unlikely to change but will need to be referenced by multiple modules, it should be added to the central constants file (name TBD). 
  * If it is a setting about pipeline behavior that may change, it should be implemented by a config file (examples are location of the calibration database and whether to save individual error terms in the output FITS files).
 
* Where should I store computed variables so they can be referenced later in the pipeline?
  * If possible, in the header of the dataset being processed or in a hew HDU extension
  * If not possible, then let's chat!
 
* Where do I save FITS files or other data files I need to use for my tests?
  * Auxiliary data to run tests should be stored in the tests/test_data folder
  * If they are larger than 1 MB, they should be stored using `git lfs`. Ask Jason about setting up git lfs (as of writing, we have not set up git lfs yet). 
