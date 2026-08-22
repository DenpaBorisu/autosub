"""
Local sherpa-onnx model management.

Downloads and verifies the FireRedASR2 CTC bilingual Chinese + English ASR
model (sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25) into a ``models/``
subfolder next to the script (or executable when frozen). The model is used by
the local (fully offline) ASR engine, outputs Chinese characters and English
words, and emits per-token timestamps for subtitle generation.
"""
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

# FireRedASR2 CTC bilingual Chinese + English ASR model (offline, int8).
# SOTA Chinese accuracy per the FireRedASR2 paper and per-token timestamps.
# ~520 MB tarball.
MODEL_NAME = "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)

# Files that must exist after extraction.
REQUIRED_FILES = {
    "model": "model.int8.onnx",
    "tokens": "tokens.txt",
}

# Silero VAD ONNX model (~2 MB). Used to split long audio into speech segments
# before feeding each to the (non-streaming) FireRedASR2 CTC recognizer, which
# would otherwise blow up memory on multi-hour files.
VAD_FILENAME = "silero_vad.onnx"
VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)

DOWNLOAD_CHUNK_SIZE = 64 * 1024


def _app_dir() -> Path:
    """Directory where the models/ folder should live."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def models_dir() -> Path:
    return _app_dir() / "models"


def model_dir() -> Path:
    return models_dir() / MODEL_NAME


def is_model_downloaded() -> bool:
    """True if all required model files (including the VAD) are present.

    The VAD is counted here so the GUI gate matches what transcription
    actually needs — otherwise a local run could be started and then fail
    mid-flight with "VAD model missing".
    """
    for _, filename in REQUIRED_FILES.items():
        if not (model_dir() / filename).is_file():
            return False
    return is_vad_downloaded()


def vad_model_path() -> str:
    """Return the absolute path of the silero VAD ONNX model."""
    path = model_dir() / VAD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing VAD model file: {path.name}")
    return str(path)


def is_vad_downloaded() -> bool:
    return (model_dir() / VAD_FILENAME).is_file()


def model_paths() -> Dict[str, str]:
    """Return absolute paths for the recognizer files.

    Raises FileNotFoundError if the model is not downloaded.
    """
    base = model_dir()
    paths: Dict[str, str] = {}
    for key, filename in REQUIRED_FILES.items():
        path = base / filename
        if path.is_file():
            paths[key] = str(path)
        else:
            raise FileNotFoundError(f"Missing model file: {path.name}")
    return paths


def model_status() -> str:
    """Human-readable model status for the GUI."""
    if is_model_downloaded():
        if is_vad_downloaded():
            return "Model ready"
        return "Model ready — VAD missing (re-download)"
    if (model_dir() / (MODEL_NAME + ".tar.bz2")).exists():
        return "Model downloaded — incomplete (re-download)"
    return "Not downloaded"


def _download_file(url: str, dest: Path, progress_cb: Callable[[int, int], None],
                   should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """Stream *url* to *dest*, reporting (downloaded, total) bytes via progress_cb."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "AutoSub/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(part, "wb") as f:
            while True:
                if should_cancel and should_cancel():
                    raise RuntimeError("Download cancelled")
                chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                progress_cb(downloaded, total)
    part.replace(dest)


def download_model(progress_cb: Callable[[int, int], None],
                   should_cancel: Optional[Callable[[], bool]] = None) -> Path:
    """Download and extract the model into models/. Returns the model directory.

    Raises RuntimeError on failure. On cancellation the partial files are removed.
    """
    target = model_dir()
    archive = models_dir() / f"{MODEL_NAME}.tar.bz2"
    try:
        _download_file(MODEL_URL, archive, progress_cb, should_cancel)
        with tarfile.open(archive, mode="r:bz2") as tf:
            tf.extractall(models_dir(), filter="data")
        for _, filename in REQUIRED_FILES.items():
            if not (target / filename).is_file():
                raise RuntimeError(f"Model archive is missing expected file: {filename}")
        _download_file(VAD_URL, target / VAD_FILENAME, progress_cb, should_cancel)
        return target
    except Exception:
        # Clean up partial downloads so a cancelled/failed attempt doesn't
        # masquerade as a valid model.
        if archive.exists():
            try:
                archive.unlink()
            except OSError:
                pass
        for suffix in (".part",):
            for path in models_dir().glob(f"{MODEL_NAME}.tar.bz2{suffix}"):
                try:
                    path.unlink()
                except OSError:
                    pass
            for path in models_dir().glob(f"{MODEL_NAME}/{VAD_FILENAME}{suffix}"):
                try:
                    path.unlink()
                except OSError:
                    pass
        raise
    finally:
        if archive.exists():
            try:
                archive.unlink()
            except OSError:
                pass


def delete_model() -> None:
    """Remove a downloaded model from disk (best-effort)."""
    target = model_dir()
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
