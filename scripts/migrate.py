import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from memory_service import MemoryService  # noqa: E402
from profile_service import ProfileService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local schema migrations")
    parser.add_argument("--sqlite", default="./data/memory.db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    MemoryService(args.sqlite)
    ProfileService(args.sqlite, default_self_interests=[])
    logging.info("Migrations applied to %s", args.sqlite)


if __name__ == "__main__":
    main()
