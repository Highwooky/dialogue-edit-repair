#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue Edit Repair App (macOS offline-safe GUI v2)

핵심 변경점:
- soundfile / dialogue_edit_repair_mvp 를 앱 시작 시 즉시 import 하지 않음
- GUI는 먼저 뜨고, 오디오 열기/저장/분석 시점에만 백엔드 의존성 확인
- libsndfile 누락 시 친절한 오류 메시지 제공
- 간단한 자체 테스트 추가 (--self-test)

전제:
- macOS 15.4 Sequoia / Apple Silicon / 폐쇄망 기준
- 복잡한 AI/torch 없이 순수 DSP + PyQt6 기반
- 먼저 '안정적으로 실행되는 도구'를 목표로 함

필수 파일:
- dialogue_edit_repair_mvp.py  (같은 폴더에 위치)

설치:
    pip install PyQt6 pyqtgraph numpy scipy soundfile

실행:
    python dialogue_edit_repair_app.py

자체 테스트:
    python dialogue_edit_repair_app.py --self-test

주의:
- 1차 버전은 미리듣기(audio playback) 없이 진행
- soundfile / libsndfile가 없으면 앱은 실행되지만 WAV 열기/저장은 동작하지 않음
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import traceback
import unittest
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "Dialogue Edit Repair"
DEV_FOOTER = "JTBC Mediatech • Production J Division • Post Production Team • Yu Byungwook"


DARK_QSS = """
QWidget {
    background-color: #0b1220;
    color: #e5eefc;
    font-size: 13px;
}
QMainWindow {
    background-color: #0b1220;
}
QGroupBox {
    border: 1px solid #25406e;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #7dd3fc;
}
QPushButton {
    background-color: #123056;
    border: 1px solid #2f5fa8;
    border-radius: 10px;
    padding: 8px 12px;
    font-weight: 700;
}
QPushButton:hover {
    background-color: #18406f;
}
QPushButton:pressed {
    background-color: #0e2744;
}
QLineEdit, QDoubleSpinBox, QTextEdit, QTableWidget {
    background-color: #0f1a2b;
    border: 1px solid #27466f;
    border-radius: 8px;
    selection-background-color: #2563eb;
}
QHeaderView::section {
    background-color: #13233b;
    color: #dbeafe;
    border: 0;
    padding: 6px;
    font-weight: 700;
}
QTableWidget {
    gridline-color: #223655;
}
QTableWidget::item:selected {
    background-color: #1d4ed8;
    color: #ffffff;
}
QCheckBox {
    spacing: 8px;
}
QLabel#titleLabel {
    font-size: 24px;
    font-weight: 800;
    color: #7dd3fc;
}
QLabel#subLabel {
    color: #9fb7d9;
}
QLabel#footerLabel {
    color: #7b8ba7;
    font-size: 11px;
}
"""


BACKEND_IMPORT_ERROR_HELP = (
    "soundfile 또는 libsndfile 백엔드를 불러오지 못했습니다.\n\n"
    "현재 앱은 GUI는 실행되지만 WAV 열기/저장/분석은 사용할 수 없습니다.\n\n"
    "확인 사항:\n"
    "1. Python 패키지 soundfile 설치 여부\n"
    "2. 시스템/번들에 libsndfile 포함 여부\n"
    "3. 폐쇄망 배포 시 libsndfile 동봉 여부\n\n"
    "원본 오류:\n{error}"
)


def decimate_waveform(audio: np.ndarray, sr: int, max_points: int = 120000) -> Tuple[np.ndarray, np.ndarray]:
    mono = ensure_mono(audio)
    if mono.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if mono.size > max_points:
        step = max(1, mono.size // max_points)
        preview = mono[::step]
        times = np.arange(preview.size, dtype=np.float64) * (step / float(sr))
    else:
        preview = mono.astype(np.float64, copy=False)
        times = np.arange(preview.size, dtype=np.float64) / float(sr)
    return times, preview


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    if audio.ndim == 2:
        return np.mean(audio, axis=1, dtype=np.float64)
    raise ValueError(f"Unsupported audio ndim: {audio.ndim}")


class LazyBackend:
    def __init__(self) -> None:
        self.sf: Any = None
        self.mvp: Any = None
        self.last_error: Optional[Exception] = None

    def is_loaded(self) -> bool:
        return self.sf is not None and self.mvp is not None

    def load(self) -> Tuple[bool, str]:
        if self.is_loaded():
            return True, ""
        try:
            sf = importlib.import_module("soundfile")
            mvp = importlib.import_module("dialogue_edit_repair_mvp")
            self.sf = sf
            self.mvp = mvp
            self.last_error = None
            return True, ""
        except Exception as exc:
            self.last_error = exc
            return False, BACKEND_IMPORT_ERROR_HELP.format(error=str(exc))

    def require(self) -> None:
        ok, message = self.load()
        if not ok:
            raise RuntimeError(message)


class WaveformWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground("#0b1220")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=False)
        self.hideButtons()
        self.plotItem.setLabel("left", "Amplitude")
        self.plotItem.setLabel("bottom", "Time", units="s")
        self.curve = self.plot(pen=pg.mkPen(width=1))
        self.event_scatter = pg.ScatterPlotItem(size=8)
        self.addItem(self.event_scatter)
        self.current_line = pg.InfiniteLine(angle=90, movable=False)
        self.addItem(self.current_line)
        self.current_line.hide()
        self._times = np.array([], dtype=np.float64)
        self._audio = np.array([], dtype=np.float64)

    def set_audio(self, audio: np.ndarray, sr: int) -> None:
        times, preview = decimate_waveform(audio, sr)
        self._times = times
        self._audio = preview
        self.curve.setData(times, preview)
        self.event_scatter.setData([], [])
        if preview.size:
            self.autoRange()

    def clear_audio(self) -> None:
        self._times = np.array([], dtype=np.float64)
        self._audio = np.array([], dtype=np.float64)
        self.curve.setData([], [])
        self.event_scatter.setData([], [])
        self.current_line.hide()

    def set_events(self, events: Sequence[Any]) -> None:
        if not self._times.size:
            self.event_scatter.setData([], [])
            return
        xs = [float(getattr(ev, "time_sec", 0.0)) for ev in events]
        ys = []
        for x in xs:
            idx = int(np.searchsorted(self._times, x))
            idx = max(0, min(self._audio.size - 1, idx))
            ys.append(float(self._audio[idx]))
        brushes = [
            pg.mkBrush("#22c55e") if getattr(ev, "decision", "") == "repair" else pg.mkBrush("#f59e0b")
            for ev in events
        ]
        self.event_scatter.setData(xs, ys, brush=brushes, pen=None)

    def focus_time(self, time_sec: float, width_sec: float = 0.08) -> None:
        self.current_line.setValue(time_sec)
        self.current_line.show()
        self.setXRange(max(0.0, time_sec - width_sec), time_sec + width_sec, padding=0.02)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 900)
        self.backend = LazyBackend()
        self.audio: Optional[np.ndarray] = None
        self.repaired_audio: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.input_path: str = ""
        self.markers_path: str = ""
        self.events: List[Any] = []
        self.cfg: Any = None
        self._build_ui()
        self._build_menu()
        self.setStyleSheet(DARK_QSS)
        self._announce_backend_status()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)
        title = QLabel(APP_TITLE)
        title.setObjectName("titleLabel")
        sub = QLabel("Offline-safe dialogue edit click repair for macOS Apple Silicon")
        sub.setObjectName("subLabel")
        main_layout.addWidget(title)
        main_layout.addWidget(sub)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        file_group = QGroupBox("Files")
        file_form = QFormLayout(file_group)
        self.input_edit = QLineEdit()
        self.markers_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("자동 생성 또는 직접 지정")
        btn_open_audio = QPushButton("Open WAV")
        btn_open_audio.clicked.connect(self.open_audio)
        btn_open_markers = QPushButton("Open Markers")
        btn_open_markers.clicked.connect(self.open_markers)
        btn_browse_output = QPushButton("Set Output")
        btn_browse_output.clicked.connect(self.choose_output)
        row1 = QWidget(); row1l = QHBoxLayout(row1); row1l.setContentsMargins(0, 0, 0, 0); row1l.addWidget(self.input_edit); row1l.addWidget(btn_open_audio)
        row2 = QWidget(); row2l = QHBoxLayout(row2); row2l.setContentsMargins(0, 0, 0, 0); row2l.addWidget(self.markers_edit); row2l.addWidget(btn_open_markers)
        row3 = QWidget(); row3l = QHBoxLayout(row3); row3l.setContentsMargins(0, 0, 0, 0); row3l.addWidget(self.output_edit); row3l.addWidget(btn_browse_output)
        file_form.addRow("Input", row1)
        file_form.addRow("Markers", row2)
        file_form.addRow("Output", row3)
        left_layout.addWidget(file_group)

        options_group = QGroupBox("Options")
        options_form = QFormLayout(options_group)
        self.spin_sensitivity = QDoubleSpinBox(); self.spin_sensitivity.setRange(0.1, 5.0); self.spin_sensitivity.setSingleStep(0.1); self.spin_sensitivity.setValue(1.0)
        self.spin_repair_ms = QDoubleSpinBox(); self.spin_repair_ms.setRange(0.1, 5.0); self.spin_repair_ms.setSingleStep(0.1); self.spin_repair_ms.setValue(0.6)
        self.spin_threshold = QDoubleSpinBox(); self.spin_threshold.setRange(0.1, 2.0); self.spin_threshold.setSingleStep(0.05); self.spin_threshold.setValue(0.85)
        self.chk_auto = QCheckBox("Enable auto detect")
        self.chk_auto.setChecked(True)
        self.chk_clap = QCheckBox("Protect clap / impact sounds")
        self.chk_clap.setChecked(True)
        self.chk_transient = QCheckBox("Protect general transients")
        self.chk_transient.setChecked(True)
        options_form.addRow("Sensitivity", self.spin_sensitivity)
        options_form.addRow("Repair half width (ms)", self.spin_repair_ms)
        options_form.addRow("Click threshold", self.spin_threshold)
        options_form.addRow("Auto detect", self.chk_auto)
        options_form.addRow("Clap protect", self.chk_clap)
        options_form.addRow("Transient protect", self.chk_transient)
        left_layout.addWidget(options_group)

        run_group = QGroupBox("Run")
        run_layout = QHBoxLayout(run_group)
        self.btn_analyze = QPushButton("Analyze")
        self.btn_repair = QPushButton("Repair + Save")
        self.btn_export_report = QPushButton("Save Report CSV")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_repair.clicked.connect(self.repair_and_save)
        self.btn_export_report.clicked.connect(self.export_report)
        run_layout.addWidget(self.btn_analyze)
        run_layout.addWidget(self.btn_repair)
        run_layout.addWidget(self.btn_export_report)
        left_layout.addWidget(run_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        left_layout.addWidget(log_group, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        wave_group = QGroupBox("Waveform")
        wave_layout = QVBoxLayout(wave_group)
        self.wave = WaveformWidget()
        wave_layout.addWidget(self.wave)
        right_layout.addWidget(wave_group, 2)
        table_group = QGroupBox("Detected Events")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Time", "Click", "Clap", "Transient", "Duration(ms)", "Decision"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        table_layout.addWidget(self.table)
        right_layout.addWidget(table_group, 2)
        splitter.addWidget(right)
        splitter.setSizes([480, 1020])
        footer = QLabel(DEV_FOOTER)
        footer.setObjectName("footerLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(footer)

    def _build_menu(self) -> None:
        action_quit = QAction("Quit", self)
        action_quit.triggered.connect(self.close)
        action_about = QAction("About", self)
        action_about.triggered.connect(self.show_about)
        action_check_backend = QAction("Check Audio Backend", self)
        action_check_backend.triggered.connect(self.show_backend_status)
        menu_file = self.menuBar().addMenu("File")
        menu_file.addAction(action_quit)
        menu_tools = self.menuBar().addMenu("Tools")
        menu_tools.addAction(action_check_backend)
        menu_help = self.menuBar().addMenu("Help")
        menu_help.addAction(action_about)

    def _announce_backend_status(self) -> None:
        ok, message = self.backend.load()
        if ok:
            self.log("[READY] audio backend loaded")
        else:
            self.log("[WARN] audio backend unavailable")
            self.log(message)

    def show_backend_status(self) -> None:
        ok, message = self.backend.load()
        if ok:
            self.show_info("Audio Backend", "soundfile / dialogue_edit_repair_mvp backend is available.")
        else:
            self.show_error("Audio Backend Unavailable", message)

    def log(self, text: str) -> None:
        self.log_box.append(text)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.log(f"[ERROR] {title}: {message}")

    def show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)
        self.log(f"[INFO] {title}: {message}")

    def require_backend(self) -> bool:
        ok, message = self.backend.load()
        if ok:
            return True
        self.show_error("Audio Backend Unavailable", message)
        return False

    def current_config(self) -> Any:
        if not self.require_backend():
            return None
        return self.backend.mvp.RepairConfig(
            auto_detect=self.chk_auto.isChecked(),
            sensitivity=float(self.spin_sensitivity.value()),
            repair_half_ms=float(self.spin_repair_ms.value()),
            clip_score_threshold=float(self.spin_threshold.value()),
            protect_claps=self.chk_clap.isChecked(),
            transient_protect=self.chk_transient.isChecked(),
        )

    def ensure_output_path(self) -> str:
        output = self.output_edit.text().strip()
        if output:
            return output
        if not self.input_path:
            return ""
        p = Path(self.input_path)
        out = p.with_name(f"{p.stem}_repaired.wav")
        self.output_edit.setText(str(out))
        return str(out)

    def open_audio(self) -> None:
        if not self.require_backend():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open WAV file", "", "Audio Files (*.wav *.aif *.aiff)")
        if not path:
            return
        try:
            audio, sr = self.backend.sf.read(path, always_2d=False)
            self.audio = audio
            self.repaired_audio = None
            self.sr = int(sr)
            self.input_path = path
            self.input_edit.setText(path)
            self.ensure_output_path()
            self.wave.set_audio(audio, self.sr)
            self.wave.set_events([])
            self.events = []
            self.table.setRowCount(0)
            self.log(f"Loaded audio: {path}")
            self.log(f"Sample rate: {sr} Hz")
            self.log(f"Channels: {1 if audio.ndim == 1 else audio.shape[1]}")
        except Exception as e:
            self.show_error("Open Audio Failed", f"{e}\n\n{traceback.format_exc()}")

    def open_markers(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open markers text", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        self.markers_path = path
        self.markers_edit.setText(path)
        self.log(f"Loaded markers: {path}")

    def choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Set output file", self.output_edit.text().strip() or "repaired.wav", "WAV Files (*.wav)")
        if not path:
            return
        self.output_edit.setText(path)

    def analyze(self) -> None:
        if self.audio is None or self.sr is None:
            self.show_error("Analyze Failed", "먼저 WAV 파일을 열어야 합니다.")
            return
        cfg = self.current_config()
        if cfg is None:
            return
        try:
            self.cfg = cfg
            mvp = self.backend.mvp
            mono = mvp.safe_mono(self.audio)
            candidate_sets = []
            if self.markers_path:
                markers_sec = mvp.read_markers(self.markers_path)
                marker_candidates = mvp.marker_collect_candidates(mono, self.sr, markers_sec, self.cfg)
                candidate_sets.append(marker_candidates)
                self.log(f"Marker-based candidates: {len(marker_candidates)}")
            if self.cfg.auto_detect or not self.markers_path:
                auto_candidates = mvp.auto_collect_candidates(mono, self.sr, self.cfg)
                candidate_sets.append(auto_candidates)
                self.log(f"Auto-detected candidates: {len(auto_candidates)}")
            if candidate_sets:
                candidates = np.unique(np.concatenate(candidate_sets))
                candidates = mvp.merge_close_indices(candidates, mvp.ms_to_samples(self.cfg.min_separation_ms, self.sr))
            else:
                candidates = np.asarray([], dtype=np.int64)
            self.events = mvp.evaluate_candidates(mono, self.sr, candidates, self.cfg)
            self.populate_table()
            self.wave.set_events(self.events)
            repair_count = sum(1 for e in self.events if getattr(e, "decision", "") == "repair")
            skip_count = len(self.events) - repair_count
            self.log(f"Total events: {len(self.events)} / Repair: {repair_count} / Skip: {skip_count}")
            self.show_info("Analyze Complete", f"Detected events: {len(self.events)}\nRepair: {repair_count}\nSkip: {skip_count}")
        except Exception as e:
            self.show_error("Analyze Failed", f"{e}\n\n{traceback.format_exc()}")

    def populate_table(self) -> None:
        self.table.setRowCount(len(self.events))
        repair_brush = QBrush(QColor("#22c55e"))
        skip_brush = QBrush(QColor("#f59e0b"))
        for row, ev in enumerate(self.events):
            values = [
                f"{float(getattr(ev, 'time_sec', 0.0)):.6f}",
                f"{float(getattr(ev, 'click_score', 0.0)):.3f}",
                f"{float(getattr(ev, 'clap_score', 0.0)):.3f}",
                f"{float(getattr(ev, 'transient_score', 0.0)):.3f}",
                f"{float(getattr(ev, 'duration_ms', 0.0)):.3f}",
                str(getattr(ev, 'decision', '')),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 5:
                    item.setForeground(repair_brush if getattr(ev, "decision", "") == "repair" else skip_brush)
                self.table.setItem(row, col, item)

    def on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.events):
            self.wave.focus_time(float(getattr(self.events[row], "time_sec", 0.0)))

    def repair_and_save(self) -> None:
        if self.audio is None or self.sr is None:
            self.show_error("Repair Failed", "먼저 WAV 파일을 열어야 합니다.")
            return
        if not self.events:
            self.show_error("Repair Failed", "먼저 Analyze를 실행해야 합니다.")
            return
        if not self.require_backend():
            return
        try:
            output_path = self.ensure_output_path()
            if not output_path:
                self.show_error("Repair Failed", "출력 경로를 지정해야 합니다.")
                return
            self.repaired_audio = self.backend.mvp.apply_repairs(self.audio, self.sr, self.events, self.cfg)
            self.backend.sf.write(output_path, self.repaired_audio, self.sr)
            repair_count = sum(1 for e in self.events if getattr(e, "decision", "") == "repair")
            self.log(f"Saved repaired file: {output_path}")
            self.show_info("Repair Complete", f"Saved: {output_path}\nApplied repairs: {repair_count}")
        except Exception as e:
            self.show_error("Repair Failed", f"{e}\n\n{traceback.format_exc()}")

    def export_report(self) -> None:
        if not self.events:
            self.show_error("Export Failed", "먼저 Analyze를 실행해야 합니다.")
            return
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV report", "repair_report.csv", "CSV Files (*.csv)")
            if not path:
                return
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["time_sec", "click_score", "clap_score", "transient_score", "duration_ms", "decision"])
                for ev in self.events:
                    writer.writerow([
                        getattr(ev, "time_sec", 0.0),
                        getattr(ev, "click_score", 0.0),
                        getattr(ev, "clap_score", 0.0),
                        getattr(ev, "transient_score", 0.0),
                        getattr(ev, "duration_ms", 0.0),
                        getattr(ev, "decision", ""),
                    ])
            self.log(f"Saved report: {path}")
            self.show_info("Export Complete", f"Saved report: {path}")
        except Exception as e:
            self.show_error("Export Failed", f"{e}\n\n{traceback.format_exc()}")

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "Dialogue Edit Repair\n\n"
            "Purpose:\n"
            "Repair short edit-point clicks in dialogue while protecting clap-like impacts.\n\n"
            "Designed for offline-safe macOS Apple Silicon workflow.\n\n"
            "This app now starts even when libsndfile is missing."
        )


class DependencyAndPreviewTests(unittest.TestCase):
    def test_ensure_mono_1d_keeps_shape(self) -> None:
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        mono = ensure_mono(x)
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(mono.shape[0], 3)
        self.assertAlmostEqual(float(mono[1]), 1.0)

    def test_ensure_mono_2d_averages_channels(self) -> None:
        x = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        mono = ensure_mono(x)
        self.assertEqual(mono.shape, (2,))
        self.assertAlmostEqual(float(mono[0]), 0.0)
        self.assertAlmostEqual(float(mono[1]), 0.5)

    def test_decimate_waveform_limits_points(self) -> None:
        x = np.linspace(-1.0, 1.0, 500_000, dtype=np.float32)
        times, preview = decimate_waveform(x, sr=48000, max_points=1000)
        self.assertLessEqual(preview.size, 1000)
        self.assertEqual(times.size, preview.size)
        self.assertGreater(preview.size, 0)

    def test_lazy_backend_reports_missing_dependency(self) -> None:
        backend = LazyBackend()
        original_import_module = importlib.import_module

        def fake_import(name: str) -> Any:
            if name == "soundfile":
                raise OSError("sndfile library not found using ctypes.util.find_library")
            return original_import_module(name)

        try:
            importlib.import_module = fake_import  # type: ignore[assignment]
            ok, message = backend.load()
        finally:
            importlib.import_module = original_import_module  # type: ignore[assignment]

        self.assertFalse(ok)
        self.assertIn("libsndfile", message)
        self.assertIn("sndfile library not found", message)


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DependencyAndPreviewTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--self-test", action="store_true", help="run lightweight self tests and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_tests()
    pg.setConfigOptions(antialias=False)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
