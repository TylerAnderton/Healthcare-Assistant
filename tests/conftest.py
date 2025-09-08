import os
import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path so `import app...` works when running pytest from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load environment variables if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure PROCESSED_DIR defaults to repo's data/processed if not set
os.environ.setdefault("PROCESSED_DIR", str(ROOT / "data/processed"))


# Pytest configuration: enable chat tests by default; allow disabling with --no-chat
def pytest_addoption(parser):
    parser.addoption(
        "--no-chat",
        action="store_true",
        default=False,
        help="Disable chat-level tests (enabled by default)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "chat: marks tests that require chat LLM + vector index")


def pytest_collection_modifyitems(config, items):
    if not items:
        return
    if config.getoption("--no-chat"):
        skip_chat = pytest.mark.skip(reason="chat tests disabled via --no-chat")
        for item in items:
            if item.get_closest_marker("chat") is not None:
                item.add_marker(skip_chat)
