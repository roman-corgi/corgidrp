# corgidrp: CORonaGraph Instrument Data Reduction Pipeline
This is the data reduction pipeline for the Nancy Grace Roman Space Telescope Coronagraph Instrument

## Install
As the code is very much still in development, clone this repository, enter the top-level folder, and run the following command:
```
pip install -e .
```
Then you can import `corgidrp` like any other python package!

## How to Contribute

Contact Jason is you have any questions on how to get started. Below is a quick tutorial. 

### The basis of getting setup
#### Find a task to work on
Check out the [Github issues page](https://github.com/roman-corgi/corgidrp/issues) for tasks that need attention. Alternatively, contact Jason (@semaphoreP).

#### Clone the git repository and install

See install instructions above. Contact Jason (@semaphoreP) if you need write access to push changes as you make edits.

##### Make a new git branch to work on

You will create a "feature branch" so you can develop your feature without impacting other people's code. Let's say I'm working on dark subtraction. I could create a feature branch and switch to it like this

```
> git branch dark-sub
> git checkout dark-sub
```

### Write your pipeline step

In `corgidrp`, each pipeline step is a function. All function should follow this example:

```
def example_step(dataset, calib_data, tuneable_arg=1, another_arg="test"):
    """
    Function docstrings are required and should follow Google style docstrings. We will not demonstrate it here for brevity.
    """
    # unless you don't alter the input dataset at all, plan to make a copy of the data
    # this is to ensure functions are reproducible
    processed_dataset = input_dataset.copy()

    ### Your code here that does the real work
    # here is a convience field to grab all the data in a dataset
    all_data = processed_dataset.all_data

    # update the header of the new dataset with your processing step
    history_msg = "I did an example step"
    # update the output dataset with the new data and update the history
    processed_dataset.update_after_processing_step(history_msg, new_all_data=all_data)

    # return the processed data
    return processed_dataset
```

Inside the function can be nearly anything you want, but the function signature and start/end of the function should follow a few rules.

  * The input dataset should always be the first input
  * Additional arguments and keywords exist only if you need them. A pipeline step can only have a single argument (the input dataset) if needed.
  * All additional function arguments/keywords should only consist of the following types: int, float, str, or a class defined in kpicdrp.Data. 
    * (Long explaination for the curious: The reason for this is that pipeline steps can be written out as text files. Int/float/str are easily represented succinctly by textfiles. All classes in kpcidrp.Data can be created simply by passing in a filepath. Therefore, all pipeline steps have easily recordably arguments for easy reproducibility.)
  * The first line of the function generally should be creating a copy of the dataset (which will be the output dataset). This way, the output dataset is not the same instance as the input dataset. This will make it easier to ensure reproducibility. 
  * The function should always end with updating the header and (typically) the data of the output dataset. The history of running this pipeline step should be recorded in the header. 

You can check out `kpicdrp.detector.dark_subtraction` function as an example. 

### Write a unit test to debug your pipeline step

We are required to write tests to verify the functionality of the code. Instead of seeing this as an extra chore, I encourage you to write the unit test to be your debug script to get your code working (this is called "test-driven development"). 

All tests are stored in the `tests` folder and each test is a function that starts with `test_`. See `tests/test_dark_sub.py` as an example. Within each test, you will likely need to simulate some mock data, run it through your function you wrote, and verify it ran correctly using assert statements. 

These tests will allow code reviews to test and understand your code. We will also run these tests in an automated test suite (continuous integration) with the pipeline. 