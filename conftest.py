import pytest

def pytest_addoption(parser):
    parser.addoption("--tvacdata_path", action="store", default="./", help="Filepath to 'CGI TVAC Data' Folder")
    parser.addoption("--e2eoutput_path", action="store", default="./", help="Directory to Save E2E Test Outputs")
    parser.addoption(
        "--which", action="store", default="unit", help="which tests to run: unit, e2e, all", choices=("unit", "e2e", "all")
    )

@pytest.fixture
def tvacdata_path(request):
    return request.config.getoption("--tvacdata_path")

@pytest.fixture
def e2eoutput_path(request):
    return request.config.getoption("--e2eoutput_path")


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark test as end-to-end")


def pytest_collection_modifyitems(config, items):
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