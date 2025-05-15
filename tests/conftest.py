import pytest
from pathlib import Path
import sys
import importlib

# --- ensure src/ is on sys.path so pytest can import paperassetpricing ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


@pytest.fixture(autouse=True, scope="session")
def setup_project_paths():
    """Dummy fixture to ensure session-level path setup runs first."""
    # nothing to do: import-path hack is done at module import
    pass


@pytest.fixture(autouse=True)
def set_test_data_dir(monkeypatch):
    """
    For each test, point DATA_DIR at tests/data so all CSV loads pick up the fixture files.
    """
    test_data_dir = PROJECT_ROOT / "tests" / "data"

    try:
        # patch the real settings module in your package
        import paperassetpricing.settings as settings

        monkeypatch.setattr(settings, "DATA_DIR", str(test_data_dir))

        # reload any modules that imported DATA_DIR at import time
        if "paperassetpricing.connectors.local.local_loader" in sys.modules:
            importlib.reload(
                sys.modules["paperassetpricing.connectors.local.local_loader"]
            )
        if "paperassetpricing.etl.aggregator" in sys.modules:
            importlib.reload(sys.modules["paperassetpricing.etl.aggregator"])

    except ImportError:
        pytest.fail(
            "CRITICAL (conftest): could not import "
            "'paperassetpricing.settings'. Ensure that your package is in src/ "
            "and that paperassetpricing/settings.py exists."
        )
    except AttributeError:
        pytest.fail(
            "CRITICAL (conftest): 'DATA_DIR' not found in "
            "paperassetpricing.settings. Ensure DATA_DIR is defined."
        )
