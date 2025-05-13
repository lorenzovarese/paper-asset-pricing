import pytest
from pathlib import Path
import sys
import importlib  # For reloading modules if necessary

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(
    autouse=True, scope="session"
)  # Changed to session scope for one-time setup
def setup_project_paths():
    """Ensures project root is in sys.path for all tests in the session."""
    # This part is mostly handled by the top-level sys.path.insert
    pass


@pytest.fixture(autouse=True)
def set_test_data_dir(monkeypatch):
    """
    Sets core.settings.DATA_DIR to point to the 'tests/data' directory
    for the duration of each test.
    """
    test_data_dir = PROJECT_ROOT / "tests" / "data"

    try:
        # Import core.settings here, inside the fixture, to ensure we get the module object
        import core.settings

        # Monkeypatch the DATA_DIR attribute on the already imported module
        monkeypatch.setattr(core.settings, "DATA_DIR", str(test_data_dir))

        # Crucial: If other modules have already imported and cached DATA_DIR,
        # they might not see the change. We might need to reload them.
        # This is a bit heavy-handed but can be necessary.
        # Only reload modules that directly import and use DATA_DIR at module level
        # or are known to cache it.
        if "connectors.local.local_loader" in sys.modules:
            importlib.reload(sys.modules["connectors.local.local_loader"])
        if "aggregator.aggregate" in sys.modules:  # If it imports local_loader
            importlib.reload(sys.modules["aggregator.aggregate"])

        # print(f"INFO (conftest): core.settings.DATA_DIR monkeypatched to: {core.settings.DATA_DIR}")

    except ImportError:
        pytest.fail(
            "CRITICAL (conftest): core.settings module not found. "
            "Tests cannot proceed without DATA_DIR configuration. "
            "Ensure core/settings.py exists and is importable."
        )
    except AttributeError:
        pytest.fail(
            "CRITICAL (conftest): core.settings.DATA_DIR attribute not found. "
            "Ensure DATA_DIR is defined in core/settings.py."
        )
