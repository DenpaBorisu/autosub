"""
Local sherpa-onnx model management.

Downloads and verifies the sherpa-onnx bilingual Chinese + English streaming
ASR Zipformer model (sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-
2023-02-16) into a ``models/`` subfolder next to the script (or executable
when frozen). The model is used by the local (fully offline) ASR engine and
outputs Chinese characters and English words.
"""
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

# Chinese + English streaming Zipformer transducer ASR model. Outputs real
# Chinese characters (not pinyin) and English text. ~40-50 MB tarball.
MODEL_NAME = "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)

ENCODER_CHUNK = "encoder-epoch-99-avg-1"
DECODER_CHUNK = "decoder-epoch-99-avg-1"
JOINER_CHUNK = "joiner-epoch-99-avg-1"

# Files that must exist after extraction. Prefer quantized (int8) encoder/joiner,
# keep fp32 decoder (quantization does not help it much).
REQUIRED_FILES = {
    "encoder": f"{ENCODER_CHUNK}.int8.onnx",
    "decoder": f"{DECODER_CHUNK}.onnx",
    "joiner": f"{JOINER_CHUNK}.int8.onnx",
    "tokens": "tokens.txt",
}

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
    """True if all required model files are present."""
    for _, filename in REQUIRED_FILES.items():
        if not (model_dir() / filename).is_file():
            return False
    return True


def model_paths() -> Dict[str, str]:
    """Return absolute paths for the recognizer files.

    Falls back to fp32 encoder/joiner if the int8 variants are missing.
    Raises FileNotFoundError if the model is not downloaded.
    """
    base = model_dir()

    def _resolve(key: str) -> str:
        path = base / REQUIRED_FILES[key]
        if path.is_file():
            return str(path)
        # Fall back to fp32 for encoder/joiner
        if key == "encoder":
            fp32 = base / f"{ENCODER_CHUNK}.onnx"
            if fp32.is_file():
                return str(fp32)
        elif key == "joiner":
            fp32 = base / f"{JOINER_CHUNK}.onnx"
            if fp32.is_file():
                return str(fp32)
        raise FileNotFoundError(f"Missing model file: {path.name}")

    return {key: _resolve(key) for key in REQUIRED_FILES}


def model_status() -> str:
    """Human-readable model status for the GUI."""
    if is_model_downloaded():
        return "Model ready"
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
