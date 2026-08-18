"""
Shared pytest fixtures for corgidrp tests.

Session-scoped fixtures eliminate redundant data generation and significantly
improve test suite performance.
"""
import pytest
import os
import tempfile


def pytest_configure(config):
    """
    Isolate each pytest-xdist worker into its own ``.corgidrp`` config/caldb dir.

    This runs inside every worker process after it starts. ``config.workerinput``
    is populated in workers under both the ``spawn`` and ``fork`` start methods, so
    the isolation activates even when workers are forked (as in CI) and did not
    re-run corgidrp's import-time setup. We recompute corgidrp's path globals off a
    worker-specific temp dir here rather than at import time, because under ``fork``
    the controller imports corgidrp while ``PYTEST_XDIST_WORKER`` is still unset and
    the forked workers inherit those already-bound paths.

    Args:
        config: pytest Config object; has a ``workerinput`` attribute in xdist workers.
    """
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return  # controller / non-xdist run: use the real ~/.corgidrp

    import corgidrp

    worker_id = workerinput.get("workerid", "gw")
    worker_tmp = tempfile.mkdtemp(prefix=f"corgidrp_worker_{worker_id}_")
    os.environ["CORGIDRP_WORKER_TMP"] = worker_tmp
    # Re-derive config_folder, caldb_filepath and default_cal_dir from the env var
    # so this worker points at its private dir instead of the inherited ~/.corgidrp.
    corgidrp.create_config_dir()
    corgidrp.update_pipeline_settings()


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle serial marker with pytest-xdist.

    Tests marked with @pytest.mark.serial are sorted by file and line number
    to ensure they run in the correct order when using --dist loadscope.

    Note: You MUST use `pytest -n auto --dist loadscope` for serial tests to work correctly.
    The loadscope distribution ensures all tests from the same module run on the same worker
    in collection order, which is critical for tests with module-level state dependencies.

    Args:
        config: pytest Config object containing configuration values
        items: list of pytest Item objects representing collected tests
    """
    # Sort all items to ensure consistent ordering
    # Particularly important for serial tests that depend on execution order
    items.sort(key=lambda item: (item.nodeid.split("::")[0], item.location[1]))
