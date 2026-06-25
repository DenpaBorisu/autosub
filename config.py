"""
Configuration management for AutoSub.

Stores settings in config.json next to the script (or next to the
executable when frozen with PyInstaller).
"""
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


def _app_dir() -> Path:
    """Return the directory where config.json should live."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


@dataclass
class Config:
    # Output
    output_dir: str = ""
    overwrite_srt: bool = False

    # Window state
    window_width: int = 700
    window_height: int = 480
    window_maximized: bool = False

    @property
    def _config_file(self) -> Path:
        return _app_dir() / "config.json"

    def save(self) -> None:
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "Config":
        config_file = _app_dir() / "config.json"
        if not config_file.exists():
            config = cls()
            config.save()
            return config
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return cls(**filtered)
        except (json.JSONDecodeError, TypeError):
            return cls()
