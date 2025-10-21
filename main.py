"""Backwards-compatible entry point for running EVCA as a script."""

from evca.main import main


if __name__ == "__main__":  # pragma: no cover - simple delegation guard
    main()
