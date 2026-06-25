# AutoSub

Drag-and-drop audio/video files to generate subtitle (.srt) files. Supports
all common media formats and handles long files by splitting into chunks
automatically.

## Requirements

- **ffmpeg/ffprobe** — must be on your system PATH
  - Debian/Ubuntu: `sudo apt install ffmpeg`
  - Arch: `sudo pacman -S ffmpeg`
  - Fedora: `sudo dnf install ffmpeg`

## Running from source

```bash
./LINUX_RUN.sh
```

This installs [uv](https://docs.astral.sh/uv/) if needed, syncs dependencies,
and launches the GUI.

Alternatively:

```bash
uv sync
uv run python autosub_gui.py
```

## Building a Linux executable

### Option A: Nuitka (recommended — ~32 MB)

Produces a single optimized native binary with unused Qt modules stripped
and LTO enabled.

```bash
uv sync
uv run nuitka \
  --onefile \
  --enable-plugin=pyqt6 \
  --remove-output \
  --lto=yes \
  --nofollow-import-to=PyQt6.QtNetwork \
  --nofollow-import-to=PyQt6.QtSql \
  --nofollow-import-to=PyQt6.QtTest \
  --nofollow-import-to=PyQt6.QtMultimedia \
  --nofollow-import-to=PyQt6.QtSvg \
  --nofollow-import-to=PyQt6.QtSvgWidgets \
  --nofollow-import-to=PyQt6.QtQml \
  --nofollow-import-to=PyQt6.QtQuick \
  --nofollow-import-to=PyQt6.QtQuick3D \
  --nofollow-import-to=PyQt6.QtQuickWidgets \
  --nofollow-import-to=PyQt6.QtQuickControls2 \
  --nofollow-import-to=PyQt6.QtWebEngineCore \
  --nofollow-import-to=PyQt6.QtWebEngineWidgets \
  --nofollow-import-to=PyQt6.QtWebChannel \
  --nofollow-import-to=PyQt6.QtWebSockets \
  --nofollow-import-to=PyQt6.QtDesigner \
  --nofollow-import-to=PyQt6.QtPrintSupport \
  --nofollow-import-to=PyQt6.QtBluetooth \
  --nofollow-import-to=PyQt6.QtConcurrent \
  --nofollow-import-to=PyQt6.QtHelp \
  --nofollow-import-to=PyQt6.QtLocation \
  --nofollow-import-to=PyQt6.QtNfc \
  --nofollow-import-to=PyQt6.QtPositioning \
  --nofollow-import-to=PyQt6.QtSensors \
  --nofollow-import-to=PyQt6.QtSerialPort \
  --nofollow-import-to=PyQt6.QtSerialBus \
  --nofollow-import-to=PyQt6.QtTextToSpeech \
  --nofollow-import-to=PyQt6.QtXml \
  --nofollow-import-to=PyQt6.QtPdf \
  --nofollow-import-to=PyQt6.QtPdfWidgets \
  --nofollow-import-to=PyQt6.QtRemoteObjects \
  --nofollow-import-to=PyQt6.QtCharts \
  --nofollow-import-to=PyQt6.QtDataVisualization \
  --nofollow-import-to=PyQt6.QtStateMachine \
  --nofollow-import-to=PyQt6.QtShaderTools \
  --nofollow-import-to=PyQt6.QtOpenGL \
  --nofollow-import-to=PyQt6.QtOpenGLWidgets \
  --nofollow-import-to=PyQt6.QtWaylandCompositor \
  --output-dir=dist \
  --output-filename=AutoSub \
  autosub_gui.py
```

Requires a C compiler (`gcc` or `clang`). Build takes 2-5 minutes.

### Option B: PyInstaller (~80 MB)

Simpler and faster to build, but larger since it bundles the entire PyQt6
package without optimization.

```bash
uv sync
uv run pyinstaller --onefile --windowed --name AutoSub autosub_gui.py
```

The executable is placed in `dist/AutoSub` for either option.

## License

GPL-3.0. See [LICENSE](LICENSE).
