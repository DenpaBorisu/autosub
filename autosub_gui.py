#!/usr/bin/env python3
"""
AutoSub - Standalone transcription utility.

Drag-and-drop audio/video files to generate subtitle (.srt) files.
"""
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QMessageBox,
    QTextEdit, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor, QIcon

from transcribe_core import transcribe_file, get_audio_files
from config import Config

MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    ".mpg", ".mpeg", ".m4v", ".3gp", ".ts", ".m2ts", ".mpga",
}


class TranscribeWorker(QThread):
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int, int)
    file_complete = pyqtSignal(str, bool, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, files: List[Path], config: Config):
        super().__init__()
        self.files = files
        self.config = config
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        total = len(self.files)
        ok_count = 0
        fail_count = 0
        failed_paths: list[Path] = []

        for i, file_path in enumerate(self.files):
            if not self._is_running:
                self.progress.emit("Cancelled")
                break

            self.progress.emit(f"Transcribing {file_path.name} ({i+1}/{total})...")

            srt_path = file_path.with_suffix('.srt')
            if self.config.output_dir:
                srt_path = Path(self.config.output_dir) / srt_path.name

            try:
                success, msg, count = transcribe_file(
                    file_path, output_path=srt_path,
                    overwrite=self.config.overwrite_srt,
                    progress_callback=lambda m: self.progress.emit(m),
                    ffmpeg_path="ffmpeg",
                    model_id="8",
                    engine="auto",
                )
                if success:
                    ok_count += 1
                    self.file_complete.emit(file_path.name, True, f"Subtitles created ({count} lines)")
                else:
                    fail_count += 1
                    failed_paths.append(file_path)
                    self.file_complete.emit(file_path.name, False, msg)
            except Exception as e:
                fail_count += 1
                failed_paths.append(file_path)
                self.file_complete.emit(file_path.name, False, str(e))

            self.progress_percent.emit(i + 1, total)

        if not self._is_running:
            self.finished.emit(False, "Cancelled")
            return

        # Clean up .chunks directories ONLY for files that succeeded.
        # Failed files keep their chunk cache so the user can re-run
        # and resume from the chunks that were already transcribed.
        for file_path in self.files:
            if file_path in failed_paths:
                continue
            srt_path = file_path.with_suffix('.srt')
            if self.config.output_dir:
                srt_path = Path(self.config.output_dir) / srt_path.name
            chunk_dir = Path(str(srt_path) + ".chunks")
            if chunk_dir.is_dir():
                shutil.rmtree(chunk_dir, ignore_errors=True)

        parts = []
        if ok_count:
            parts.append(f"{ok_count} succeeded")
        if fail_count:
            parts.append(f"{fail_count} failed")
        msg = ", ".join(parts) + "." if parts else "No files processed."

        if failed_paths:
            names = ", ".join(p.name for p in failed_paths[:3])
            extra = f" Re-run to resume {len(failed_paths)} failed file(s) ({names})"
            if len(failed_paths) > 3:
                extra += f" and {len(failed_paths) - 3} more"
            extra += " — completed chunks are cached."
            msg += extra

        self.finished.emit(fail_count == 0, msg)


class DropListWidget(QListWidget):
    def __init__(self, parent_window, parent=None):
        super().__init__(parent)
        self.parent_window = parent_window
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.is_file():
                    self.parent_window._add_file(path)
                elif path.is_dir():
                    self.parent_window._add_folder_files(path)
            event.acceptProposedAction()


class AutoSubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config.load()
        self._processing_start_time: Optional[float] = None
        self._current_status_msg = ""

        self._setup_ui()
        self._restore_geometry()

    def _setup_ui(self):
        self.setWindowTitle("AutoSub")
        self.resize(700, 480)
        self.setMinimumSize(500, 350)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # File list
        files_label = QLabel("Files")
        files_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(files_label)

        hint = QLabel("Drop video or audio files here or click Add Files")
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        self.file_list = DropListWidget(self)
        self.file_list.addItem("Drop video or audio files here")
        item = self.file_list.item(0)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setForeground(QColor("#888"))
        main_layout.addWidget(self.file_list, 1)

        # Buttons
        btns = QHBoxLayout()
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self._add_files)
        folder_btn = QPushButton("Add Folder...")
        folder_btn.clicked.connect(self._add_folder)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_files)
        btns.addWidget(add_btn)
        btns.addWidget(folder_btn)
        btns.addStretch()
        btns.addWidget(clear_btn)
        main_layout.addLayout(btns)

        # Start / Cancel
        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedHeight(42)
        self.start_btn.setProperty("primary", True)
        self.start_btn.clicked.connect(self._start_processing)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(42)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.cancel_btn.setVisible(False)

        start_layout = QHBoxLayout()
        start_layout.addWidget(self.start_btn)
        start_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(start_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: white; font-size: 11px;")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        # Log toggle
        self.log_toggle_btn = QPushButton("Show log")
        self.log_toggle_btn.setFlat(True)
        self.log_toggle_btn.setStyleSheet("color: #aaa; font-size: 10px; padding: 2px 4px; text-align: left; border: none;")
        self.log_toggle_btn.clicked.connect(self._toggle_log)
        main_layout.addWidget(self.log_toggle_btn)

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        self.log_text.setVisible(False)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 4px;
                color: palette(text);
                font-family: monospace;
                font-size: 11px;
                padding: 4px;
            }
        """)
        main_layout.addWidget(self.log_text)

        central.setLayout(main_layout)
        self._apply_stylesheet()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget { color: palette(text); }
            QPushButton[primary="true"] {
                background-color: palette(highlight);
                color: palette(highlighted-text);
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton[primary="true"]:hover { opacity: 0.9; }
            QPushButton[primary="true"]:disabled {
                background-color: palette(mid);
                color: palette(mid);
            }
            QListWidget {
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 4px;
                color: palette(text);
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid palette(midlight);
            }
            QListWidget::item:selected {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
            QListWidget::item:hover {
                background-color: palette(midlight);
            }
            QProgressBar {
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: palette(highlight);
                border-radius: 3px;
            }
        """)

    def _restore_geometry(self):
        self.resize(self.config.window_width, self.config.window_height)
        if self.config.window_maximized:
            self.showMaximized()

    def closeEvent(self, event):
        self.config.window_width = self.width()
        self.config.window_height = self.height()
        self.config.window_maximized = self.isMaximized()
        self.config.save()
        super().closeEvent(event)

    def _is_media_file(self, path: Path) -> bool:
        return path.suffix.lower() in MEDIA_EXTENSIONS

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Files...",
            filter="Media Files (*.mp3 *.wav *.mp4 *.mkv *.avi *.mov *.webm *.flac *.m4a *.ogg);;All Files (*)")
        for file in files:
            self._add_file(Path(file))

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add Folder...")
        if folder:
            self._add_folder_files(Path(folder))

    def _add_folder_files(self, folder: Path):
        for file in get_audio_files(str(folder)):
            self._add_file(file)

    def _add_file(self, file_path: Path):
        if not self._is_media_file(file_path):
            return

        # Remove placeholder
        if self.file_list.count() == 1 and self.file_list.item(0).flags() == Qt.ItemFlag.NoItemFlags:
            self.file_list.clear()

        # Deduplicate
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == str(file_path):
                return

        item = QListWidgetItem(f"  {file_path.name}")
        item.setData(Qt.ItemDataRole.UserRole, str(file_path))
        self.file_list.addItem(item)

    def _clear_files(self):
        self.file_list.clear()
        item = QListWidgetItem("Drop video or audio files here")
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setForeground(QColor("#888"))
        self.file_list.addItem(item)

    def _mark_file_done(self, row: int, success: bool):
        if row < 0 or row >= self.file_list.count():
            return
        item = self.file_list.item(row)
        if success:
            item.setBackground(QColor("#1b3a1b"))
            item.setForeground(QColor("#4caf50"))
        else:
            item.setBackground(QColor("#3a1b1b"))
            item.setForeground(QColor("#f44336"))
        icon = QIcon.fromTheme("dialog-ok" if success else "dialog-error")
        if not icon.isNull():
            item.setIcon(icon)

    def _get_file_paths(self) -> List[Path]:
        paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.flags() == Qt.ItemFlag.NoItemFlags:
                continue
            paths.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        return paths

    def _start_processing(self):
        files = self._get_file_paths()
        if not files:
            QMessageBox.warning(self, "No files", "Add some files first!")
            return

        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._processing_start_time = time.time()
        self._current_status_msg = f"Processing {len(files)} file(s)..."
        self.status_label.setText(self._current_status_msg)

        self.worker = TranscribeWorker(files, self.config)
        self.worker.progress.connect(self._on_progress)
        self.worker.progress_percent.connect(self._update_progress)
        self.worker.file_complete.connect(self._on_file_complete)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _cancel_processing(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.terminate()
            self.worker.wait(3000)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelled")
            self._append_log("Cancelled by user")

    def _on_progress(self, msg: str):
        self._current_status_msg = msg
        self._update_status_with_eta()
        self._append_log(msg)

    def _append_log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _toggle_log(self):
        visible = self.log_text.isVisible()
        self.log_text.setVisible(not visible)
        self.log_toggle_btn.setText("Hide log" if visible else "Show log")

    def _update_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current * 100 / total))
        self._update_status_with_eta()

    def _update_status_with_eta(self):
        text = self._current_status_msg
        if self._processing_start_time and self.progress_bar.value() > 0:
            elapsed = time.time() - self._processing_start_time
            progress = self.progress_bar.value() / 100.0
            if progress > 0.01:
                remaining = elapsed / progress - elapsed
                text += f"  ·  {self._format_eta(remaining)}"
        self.status_label.setText(text)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds < 60:
            return f"~{int(seconds)}s left"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"~{m}m {s}s left"
        else:
            h, remainder = divmod(int(seconds), 3600)
            m = remainder // 60
            return f"~{h}h {m}m left"

    def _on_file_complete(self, filename: str, success: bool, msg: str):
        files = self._get_file_paths()
        for i, f in enumerate(files):
            if f.name == filename:
                self._mark_file_done(i, success)
                break

        prefix = "OK" if success else "FAIL"
        self._current_status_msg = f"{filename}: {msg}"
        self.status_label.setText(self._current_status_msg)
        self._append_log(f"[{prefix}] {filename}: {msg}")

    def _on_finished(self, success: bool, msg: str):
        self._processing_start_time = None
        self.start_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(msg if success else "Done with errors")
        if success:
            QMessageBox.information(self, "Done", msg)
        else:
            QMessageBox.warning(self, "Done with errors", msg)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AutoSub")

    window = AutoSubWindow()
    window.show()

    for file in sys.argv[1:]:
        path = Path(file)
        if path.exists():
            window._add_file(path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
