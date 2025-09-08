import json
from pathlib import Path

import pytest

from app.chains.chat import answer_question

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Mark this module as requiring chat; conftest.py enables chat tests by default
# and allows disabling with --no-chat
pytestmark = pytest.mark.chat


def load_fixture():
    with open(FIXTURES_DIR / "prompts.json", "r") as f:
        return json.load(f)


@pytest.mark.parametrize("case", load_fixture()["cases"])
def test_chat_prompts_include_expected_strings(case):
    prompt = case["prompt"]
    expect = case["expect_contains"]
    ans, sources = answer_question(prompt, history=[])
    for token in expect:
        assert token in ans, f"Expected token '{token}' not found in answer.\nAnswer was:\n{ans}"


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Run chat prompt tests")
    parser.add_argument("--no-chat", action="store_true", help="Disable chat tests (debug)")
    args = parser.parse_args(argv)
    import pytest as _pytest
    test_path = str(Path(__file__).resolve())
    opts = [test_path]
    if args.no_chat:
        opts.insert(0, "--no-chat")
    rc = _pytest.main(opts)
    sys.exit(rc)


if __name__ == "__main__":
    main()
