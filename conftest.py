"""
These functions below are pytest hooks for setting up the pytest environment
"""

import pytest

def pytest_addoption(parser):
    """
    This adds command line arguments to the pytest command used for end-to-end testing

    Args:
        parser: argument parser object
    """
    # can add path to TVAC data folder
    parser.addoption("--tvacdata_path", action="store", default="./", help="Filepath to 'CGI TVAC Data' Folder")
    # can specify output for e2e tests
    parser.addoption("--e2eoutput_path", action="store", default="./", help="Directory to Save E2E Test Outputs")
    # add ability to specify whether to run unit tests, e2e tests, or both
    parser.addoption(
        "--which", action="store", default="unit", help="which tests to run: unit, e2e, all", choices=("unit", "e2e", "all")
    )

@pytest.fixture
def tvacdata_path(request):
    """
    Adds the hook to be able to grab the value passed in with the tvacdata_path argument

    Args:
        request (FixtureRequest): pytest request of a fixture
    
    Returns:
        str: value from this command line argument
    """
    return request.config.getoption("--tvacdata_path")

@pytest.fixture
def e2eoutput_path(request):
    """
    Adds the hook to be able to grab the value passed in with the e2eoutput_path argument

    Args:
        request (FixtureRequest): pytest request of a fixture
    
    Returns:
        str: value from this command line argument
    """
    return request.config.getoption("--e2eoutput_path")


def pytest_configure(config):
    """
    Adds e2e marker for specifying e2e tests

    Args:
        config: pytest configuration object
    """
    config.addinivalue_line("markers", "e2e: mark test as end-to-end")


def pytest_collection_modifyitems(config, items):
    """
    Determines which set of tests to run, based on --which argument

    By default, runs only unit tests (those not marked as e2e)

    Args:
        config: pytest configuration object
        items: list of tests
    """
    whichtests = config.getoption("--which")
    if whichtests == "all":
        # --e2e given in cli: do not skip slow tests
        return
    elif whichtests == "unit":
        skip_e2e = pytest.mark.skip(reason="Marked as e2e")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
    elif whichtests == "e2e":
        skip_unit = pytest.mark.skip(reason="Not marked as e2e")
        for item in items:
            if not "e2e" in item.keywords:
                item.add_marker(skip_unit)