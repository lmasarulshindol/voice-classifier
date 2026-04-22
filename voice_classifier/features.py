"""音声ファイルから分類に必要な特徴量を抽出する."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    """音声から抽出した特徴量の集合.

    すべて float でまとめてあるため、そのまま辞書化やロギングに使える.
    """

    duration_sec: float
    sample_rate: int

    # 基本周波数 (F0) 関連 — 声の高さと抑揚を表す
    f0_mean_hz: float
    f0_median_hz: float
    f0_std_hz: float
    f0_min_hz: float
    f0_max_hz: float
    voiced_ratio: float  # 有声区間の割合 (0-1)

    # エネルギー関連 — 声の大きさ/勢い
    rms_mean: float
    rms_std: float

    # スペクトル特徴 — 声質/明るさ
    spectral_centroid_mean: float  # 明るい声ほど高い
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    zero_crossing_rate_mean: float  # かすれ/息の成分

    # 話速の近似指標 (オンセットの密度)
    onset_rate_per_sec: float


def _safe_mean(arr: np.ndarray) -> float:
    """NaN を除いた平均。全て NaN なら 0.0 を返す."""
    if arr.size == 0:
        return 0.0
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if valid.size > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    valid = arr[~np.isnan(arr)]
    return float(valid.std()) if valid.size > 0 else 0.0


def extract_features(
    audio_path: Union[str, Path],
    target_sr: int = 22050,
) -> AudioFeatures:
    """音声ファイルを読み込み、特徴量を抽出する.

    Args:
        audio_path: 解析対象の音声ファイルパス (wav/mp3/m4a/flac など).
        target_sr: リサンプリング後のサンプリングレート.

    Returns:
        AudioFeatures: 抽出した特徴量.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    # 音声読み込み (librosa が ffmpeg/audioread 経由で多くの形式に対応)
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"音声データが空です: {path}")

    duration = librosa.get_duration(y=y, sr=sr)

    # F0 推定 (pyin は無声区間を NaN として返す)
    # 人間の声の大まかな範囲: 50Hz〜600Hz
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),   # 約 65 Hz
        fmax=float(librosa.note_to_hz("C6")),   # 約 1047 Hz
        sr=sr,
    )

    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if voiced_f0.size > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_min = float(np.min(voiced_f0))
        f0_max = float(np.max(voiced_f0))
    else:
        f0_mean = f0_median = f0_std = f0_min = f0_max = 0.0

    voiced_ratio = (
        float(np.mean(voiced_flag.astype(float))) if voiced_flag is not None else 0.0
    )

    # RMS (音量)
    rms = librosa.feature.rms(y=y)[0]

    # スペクトル特徴
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    # オンセット検出で話速の近似値を算出
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_rate = float(len(onsets)) / duration if duration > 0 else 0.0

    return AudioFeatures(
        duration_sec=float(duration),
        sample_rate=int(sr),
        f0_mean_hz=f0_mean,
        f0_median_hz=f0_median,
        f0_std_hz=f0_std,
        f0_min_hz=f0_min,
        f0_max_hz=f0_max,
        voiced_ratio=voiced_ratio,
        rms_mean=_safe_mean(rms),
        rms_std=_safe_std(rms),
        spectral_centroid_mean=_safe_mean(spec_centroid),
        spectral_bandwidth_mean=_safe_mean(spec_bandwidth),
        spectral_rolloff_mean=_safe_mean(spec_rolloff),
        zero_crossing_rate_mean=_safe_mean(zcr),
        onset_rate_per_sec=onset_rate,
    )
