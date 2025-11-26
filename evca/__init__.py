"""Enhanced Video Complexity Analyzer (EVCA) package."""
from importlib import resources
from pathlib import Path

from .analyze import analyze_frames, EVCAConfig, EVCAResult

PACKAGE_ROOT = Path(__file__).resolve().parent

__all__ = ["PACKAGE_ROOT", "analyze_frames", "EVCAConfig", "EVCAResult"]
