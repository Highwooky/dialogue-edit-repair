#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue Edit Repair MVP

목표:
- 대사 편집점에서 발생하는 짧은 클릭/틱 소리를 탐지하고 국소 복원
- 박수(clap)나 의도된 강한 트랜지언트는 최대한 보호
- 독립 실행형 오프라인 처리 엔진 (GUI 전 단계)

설치:
    pip install numpy scipy soundfile

사용 예시:
    python dialogue_edit_repair_mvp.py \
        --input input.wav \
        --output repaired.wav \
        --report report.csv \
        --markers markers.txt

markers.txt 형식(초 단위, 한 줄 하나):
    12.340
    19.875
    25.000

마커 파일이 없으면 전체 구간 자동 탐지를 수행한다.

핵심 아이디어:
1) 편집점 중심 또는 자동 탐지로 클릭 후보 수집
2) 초단기 파형 불연속 / 고주파 burst / 국소 피크 비정상성 계산
3) 박수/타격음 보호 스코어 계산
4) 클릭만 매우 짧게 마이크로 크로스페이드 + 선형 보간 복원

주의:
- 이 버전은 '실무용 첫 엔진'으로, RX급 범용 디클릭이 아니다.
- 가장 잘 맞는 용도는 '대사 편집점 틱/툭 정리'이다.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


@dataclass
class RepairConfig:
    auto_detect: bool = True
    marker_window_ms: float = 20.0
    detect_window_ms: float = 2.5
    repair_half_ms: float = 0.6
    min_separation_ms: float = 3.0
    sensitivity: float = 1.0
    protect_claps: bool = True
    clap_protect_strength: float = 1.0
    transient_protect: bool = True
    peak_z_threshold: float = 6.0
    deriv_z_threshold: float = 5.0
    hf_z_threshold: float = 4.0
    clip_score_threshold: float = 0.85


@dataclass
class CandidateEvent:
    sample_index: int
    time_sec: float
    peak_z: float
    deriv_z: float
    hf_z: float
    asymmetry: float
    duration_ms: float
    clap_score: float
    transient_score: float
    click_score: float
    decision: str


def ms_to_samples(ms: float, sr: int) -> int:
    return max(1, int(round(ms * sr / 1000.0)))


def safe_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, win)
    pad = win // 2
    x2 = np.pad(x * x, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=np.float64) / win
    y = np.convolve(x2, kernel, mode="valid")
    return np.sqrt(np.maximum(y, 1e-15))


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = sr * 0.5
    low_n = max(1e-5, low / nyq)
    high_n = min(0.999, high / nyq)
    if high_n <= low_n:
        return x.copy()
    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def merge_close_indices(indices: np.ndarray, min_gap: int) -> np.ndarray:
    if len(indices) == 0:
        return indices
    indices = np.sort(indices)
    merged = [int(indices[0])]
    for idx in indices[1:]:
        if idx - merged[-1] >= min_gap:
            merged.append(int(idx))
    return np.asarray(merged, dtype=np.int64)


def read_markers(path: Optional[str]) -> List[float]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Marker file not found: {p}")
    markers = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        markers.append(float(line))
    return markers


def compute_detection_features(mono: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = mono.astype(np.float64)
    abs_peak_z = np.abs(robust_z(np.abs(x)))
    d1 = np.diff(x, prepend=x[0])
    deriv_z = np.abs(robust_z(np.abs(d1)))
    hf = bandpass(x, sr, 2500.0, min(12000.0, sr * 0.45), order=4)
    hf_env = moving_rms(hf, ms_to_samples(0.35, sr))
    hf_z = np.abs(robust_z(hf_env))
    local_rms = moving_rms(x, ms_to_samples(5.0, sr))
    return abs_peak_z, deriv_z, hf_z, local_rms


def estimate_event_duration(hf_env: np.ndarray, center: int, sr: int) -> float:
    n = len(hf_env)
    c = int(np.clip(center, 0, n - 1))
    peak = hf_env[c]
    if peak <= 1e-12:
        return 0.0
    thr = peak * 0.25
    l = c
    while l > 0 and hf_env[l] > thr:
        l -= 1
    r = c
    while r < n - 1 and hf_env[r] > thr:
        r += 1
    return (r - l) * 1000.0 / sr


def compute_asymmetry(x: np.ndarray, center: int, width: int) -> float:
    n = len(x)
    l0 = max(0, center - width)
    l1 = max(0, center)
    r0 = min(n, center)
    r1 = min(n, center + width)
    left = x[l0:l1]
    right = x[r0:r1]
    m = min(len(left), len(right))
    if m < 8:
        return 0.0
    left = left[-m:]
    right = right[:m]
    denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-12
    corr = float(np.dot(left, right) / denom)
    return 1.0 - max(-1.0, min(1.0, corr))


def compute_clap_score(mono: np.ndarray, sr: int, center: int) -> float:
    pre = ms_to_samples(20.0, sr)
    post = ms_to_samples(60.0, sr)
    a = max(0, center - pre)
    b = min(len(mono), center + post)
    seg = mono[a:b]
    if len(seg) < ms_to_samples(10, sr):
        return 0.0
    bp = bandpass(seg, sr, 1200.0, min(10000.0, sr * 0.45))
    env = moving_rms(bp, ms_to_samples(0.5, sr))
    if len(env) < 10:
        return 0.0
    c = center - a
    c = int(np.clip(c, 0, len(env) - 1))
    peak = float(env[c]) + 1e-12
    tail_end = min(len(env), c + ms_to_samples(18.0, sr))
    tail = env[c:tail_end]
    tail_ratio = float(np.mean(tail) / peak) if len(tail) else 0.0
    dur_ms = estimate_event_duration(env, c, sr)
    dur_score = np.clip((dur_ms - 1.5) / 8.0, 0.0, 1.0)
    peak_mask = env > (0.55 * np.max(env))
    rising = np.diff(peak_mask.astype(np.int8), prepend=0)
    peak_groups = int(np.sum(rising == 1))
    repeat_score = np.clip((peak_groups - 1) / 3.0, 0.0, 1.0)
    tail_score = np.clip((tail_ratio - 0.15) / 0.45, 0.0, 1.0)
    return float(np.clip(0.50 * dur_score + 0.35 * tail_score + 0.15 * repeat_score, 0.0, 1.0))


def compute_transient_score(mono: np.ndarray, sr: int, center: int) -> float:
    w = ms_to_samples(12.0, sr)
    a = max(0, center - w)
    b = min(len(mono), center + w)
    seg = mono[a:b]
    if len(seg) < 16:
        return 0.0
    env = moving_rms(seg, ms_to_samples(0.4, sr))
    c = center - a
    c = int(np.clip(c, 1, len(env) - 2))
    pre = float(np.mean(env[max(0, c - ms_to_samples(2.0, sr)):c]) + 1e-12)
    post = float(np.mean(env[c:min(len(env), c + ms_to_samples(6.0, sr))]) + 1e-12)
    ratio = post / pre
    score = np.clip((ratio - 1.2) / 3.0, 0.0, 1.0)
    return float(score)


def compute_click_score(
    peak_z: float,
    deriv_z: float,
    hf_z: float,
    asymmetry: float,
    duration_ms: float,
    clap_score: float,
    transient_score: float,
    cfg: RepairConfig,
) -> float:
    shortness = 1.0 - np.clip((duration_ms - 0.8) / 5.0, 0.0, 1.0)
    peak_term = np.clip(peak_z / cfg.peak_z_threshold, 0.0, 2.0)
    deriv_term = np.clip(deriv_z / cfg.deriv_z_threshold, 0.0, 2.0)
    hf_term = np.clip(hf_z / cfg.hf_z_threshold, 0.0, 2.0)
    asym_term = np.clip(asymmetry, 0.0, 1.0)
    score = (0.28 * peak_term + 0.30 * deriv_term + 0.24 * hf_term + 0.18 * asym_term)
    score *= (0.60 + 0.40 * shortness)
    if cfg.protect_claps:
        score *= (1.0 - 0.70 * cfg.clap_protect_strength * np.clip(clap_score, 0.0, 1.0))
    if cfg.transient_protect:
        score *= (1.0 - 0.40 * np.clip(transient_score, 0.0, 1.0))
    return float(np.clip(score, 0.0, 2.0))


def auto_collect_candidates(mono: np.ndarray, sr: int, cfg: RepairConfig) -> np.ndarray:
    abs_peak_z, deriv_z, hf_z, _ = compute_detection_features(mono, sr)
    peak_hits = np.where(abs_peak_z > (cfg.peak_z_threshold / max(cfg.sensitivity, 1e-6)))[0]
    deriv_hits = np.where(deriv_z > (cfg.deriv_z_threshold / max(cfg.sensitivity, 1e-6)))[0]
    hf_hits = np.where(hf_z > (cfg.hf_z_threshold / max(cfg.sensitivity, 1e-6)))[0]
    all_hits = np.unique(np.concatenate([peak_hits, deriv_hits, hf_hits]))
    min_gap = ms_to_samples(cfg.min_separation_ms, sr)
    return merge_close_indices(all_hits, min_gap)


def marker_collect_candidates(mono: np.ndarray, sr: int, markers_sec: List[float], cfg: RepairConfig) -> np.ndarray:
    if not markers_sec:
        return np.asarray([], dtype=np.int64)
    abs_peak_z, deriv_z, hf_z, _ = compute_detection_features(mono, sr)
    win = ms_to_samples(cfg.marker_window_ms, sr)
    out = []
    for t in markers_sec:
        c = int(round(t * sr))
        a = max(0, c - win)
        b = min(len(mono), c + win)
        if b - a < 8:
            continue
        combo = (
            np.clip(abs_peak_z[a:b] / cfg.peak_z_threshold, 0, None) +
            np.clip(deriv_z[a:b] / cfg.deriv_z_threshold, 0, None) +
            np.clip(hf_z[a:b] / cfg.hf_z_threshold, 0, None)
        )
        best = a + int(np.argmax(combo))
        out.append(best)
    min_gap = ms_to_samples(cfg.min_separation_ms, sr)
    return merge_close_indices(np.asarray(out, dtype=np.int64), min_gap)


def evaluate_candidates(mono: np.ndarray, sr: int, candidates: np.ndarray, cfg: RepairConfig) -> List[CandidateEvent]:
    abs_peak_z, deriv_z, hf_z, _ = compute_detection_features(mono, sr)
    hf = bandpass(mono, sr, 2500.0, min(12000.0, sr * 0.45), order=4)
    hf_env = moving_rms(hf, ms_to_samples(0.35, sr))
    events: List[CandidateEvent] = []
    asym_w = ms_to_samples(0.8, sr)
    for c in candidates:
        c = int(np.clip(c, 0, len(mono) - 1))
        pz = float(abs_peak_z[c])
        dz = float(deriv_z[c])
        hz = float(hf_z[c])
        asym = float(compute_asymmetry(mono, c, asym_w))
        dur = float(estimate_event_duration(hf_env, c, sr))
        clap_score = float(compute_clap_score(mono, sr, c))
        trans_score = float(compute_transient_score(mono, sr, c))
        click_score = float(compute_click_score(pz, dz, hz, asym, dur, clap_score, trans_score, cfg))
        decision = "repair" if click_score >= cfg.clip_score_threshold else "skip"
        events.append(CandidateEvent(c, c / sr, pz, dz, hz, asym, dur, clap_score, trans_score, click_score, decision))
    return events


def repair_click_region(channel: np.ndarray, center: int, half_width: int) -> np.ndarray:
    y = channel.copy()
    n = len(y)
    a = max(1, center - half_width)
    b = min(n - 2, center + half_width)
    if b <= a + 1:
        return y
    left = y[a - 1]
    right = y[b + 1]
    interp = np.linspace(left, right, (b - a + 1), dtype=np.float64)
    orig = y[a:b + 1].astype(np.float64)
    t = np.linspace(0.0, 1.0, len(orig), dtype=np.float64)
    blend = 0.85 - 0.25 * np.cos(2.0 * np.pi * t)
    blend = np.clip(blend, 0.55, 1.0)
    y[a:b + 1] = (1.0 - blend) * orig + blend * interp
    return y


def apply_repairs(audio: np.ndarray, sr: int, events: List[CandidateEvent], cfg: RepairConfig) -> np.ndarray:
    out = audio.copy().astype(np.float64)
    half = ms_to_samples(cfg.repair_half_ms, sr)
    for ev in events:
        if ev.decision != "repair":
            continue
        c = ev.sample_index
        if out.ndim == 1:
            out = repair_click_region(out, c, half)
        else:
            for ch in range(out.shape[1]):
                out[:, ch] = repair_click_region(out[:, ch], c, half)
    return out


def write_report(path: str, events: List[CandidateEvent]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()) if events else [
            "sample_index", "time_sec", "peak_z", "deriv_z", "hf_z", "asymmetry",
            "duration_ms", "clap_score", "transient_score", "click_score", "decision"
        ])
        writer.writeheader()
        for ev in events:
            writer.writerow(asdict(ev))


def process_file(
    input_path: str,
    output_path: str,
    report_path: Optional[str],
    markers_path: Optional[str],
    cfg: RepairConfig,
) -> Tuple[np.ndarray, int, List[CandidateEvent]]:
    audio, sr = sf.read(input_path, always_2d=False)
    mono = safe_mono(audio)
    markers_sec = read_markers(markers_path)
    candidate_sets = []
    if markers_sec:
        candidate_sets.append(marker_collect_candidates(mono, sr, markers_sec, cfg))
    if cfg.auto_detect or not markers_sec:
        candidate_sets.append(auto_collect_candidates(mono, sr, cfg))
    if candidate_sets:
        candidates = np.unique(np.concatenate(candidate_sets))
        candidates = merge_close_indices(candidates, ms_to_samples(cfg.min_separation_ms, sr))
    else:
        candidates = np.asarray([], dtype=np.int64)
    events = evaluate_candidates(mono, sr, candidates, cfg)
    repaired = apply_repairs(audio, sr, events, cfg)
    sf.write(output_path, repaired, sr)
    if report_path:
        write_report(report_path, events)
    return repaired, sr, events


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dialogue edit click repair MVP")
    p.add_argument("--input", required=True, help="입력 WAV/AIFF 파일")
    p.add_argument("--output", required=True, help="출력 WAV 파일")
    p.add_argument("--report", default="", help="CSV 리포트 저장 경로")
    p.add_argument("--markers", default="", help="초 단위 편집점 목록 txt 파일")
    p.add_argument("--no-auto-detect", action="store_true", help="자동 탐지 비활성화")
    p.add_argument("--sensitivity", type=float, default=1.0, help="탐지 민감도")
    p.add_argument("--repair-half-ms", type=float, default=0.6, help="복원 반폭(ms)")
    p.add_argument("--clip-threshold", type=float, default=0.85, help="최종 클릭 판정 임계치")
    p.add_argument("--disable-clap-protect", action="store_true", help="박수 보호 비활성화")
    p.add_argument("--disable-transient-protect", action="store_true", help="일반 트랜지언트 보호 비활성화")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = RepairConfig(
        auto_detect=not args.no_auto_detect,
        sensitivity=max(0.1, float(args.sensitivity)),
        repair_half_ms=max(0.1, float(args.repair_half_ms)),
        clip_score_threshold=max(0.1, float(args.clip_threshold)),
        protect_claps=not args.disable_clap_protect,
        transient_protect=not args.disable_transient_protect,
    )
    _, sr, events = process_file(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report or None,
        markers_path=args.markers or None,
        cfg=cfg,
    )
    repaired_count = sum(1 for e in events if e.decision == "repair")
    skipped_count = sum(1 for e in events if e.decision != "repair")
    print("=" * 60)
    print("Dialogue Edit Repair MVP 완료")
    print(f"Sample Rate   : {sr} Hz")
    print(f"Candidates    : {len(events)}")
    print(f"Repaired      : {repaired_count}")
    print(f"Skipped       : {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
