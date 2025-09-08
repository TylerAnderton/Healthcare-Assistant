import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when running directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run_module_main(mod, argv=None) -> int:
    try:
        mod.main(argv or [])
        return 0  # main() may not exit; treat as success
    except SystemExit as e:
        # main() uses sys.exit(rc)
        return int(e.code) if isinstance(e.code, int) else (0 if not e.code else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all Healthcare Assistant tests")
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Disable chat-level tests (enabled by default)",
    )
    args = parser.parse_args(argv)

    # Import test modules lazily
    from tests import test_labs_accuracy as t_labs
    from tests import test_meds_accuracy as t_meds
    from tests import test_whoop_accuracy as t_whoop
    from tests import test_chat_prompts as t_chat

    rc_all = 0
    rc_all |= _run_module_main(t_labs, [])
    rc_all |= _run_module_main(t_meds, [])
    rc_all |= _run_module_main(t_whoop, [])
    chat_argv = ["--no-chat"] if args.no_chat else []
    rc_all |= _run_module_main(t_chat, chat_argv)

    sys.exit(rc_all)


if __name__ == "__main__":
    main()
