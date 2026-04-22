"""特徴量から声のタイプを分類するヒューリスティック群.

いずれも事前学習モデルを使わず、音響特徴量からのルールベース推定.
一般的な音声学の目安値をもとに設計しているが、あくまで簡易判定なので
厳密な識別には機械学習ベースの手法が必要.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .features import AudioFeatures


Gender = Literal["male", "female", "unknown"]
AgeGroup = Literal["child", "young_adult", "adult", "senior", "unknown"]
PitchLevel = Literal["very_low", "low", "mid", "high", "very_high"]
VoiceType = Literal[
    "bass", "baritone", "tenor", "countertenor", "alto", "mezzo_soprano", "soprano", "unknown"
]
Emotion = Literal["calm", "happy", "excited", "angry", "sad", "neutral"]


@dataclass(frozen=True)
class Classification:
    """分類結果 1 件分. label と信頼度, 説明文を保持する."""

    label: str
    confidence: float  # 0.0 - 1.0
    detail: str


def _confidence_from_distance(value: float, center: float, half_width: float) -> float:
    """中心値からの距離に応じて 0.5-1.0 の信頼度を返す簡易関数."""
    if half_width <= 0:
        return 0.5
    dist = abs(value - center)
    if dist >= half_width:
        return 0.5
    return 1.0 - 0.5 * (dist / half_width)


def classify_pitch(features: AudioFeatures) -> Classification:
    """F0 の中央値から声の高さを 5 段階で分類."""
    f0 = features.f0_median_hz

    if f0 <= 0:
        return Classification("unknown", 0.0, "有声区間が検出できませんでした")

    if f0 < 100:
        label: PitchLevel = "very_low"
    elif f0 < 160:
        label = "low"
    elif f0 < 220:
        label = "mid"
    elif f0 < 300:
        label = "high"
    else:
        label = "very_high"

    confidence = _confidence_from_distance(f0, center=f0, half_width=40.0)
    detail = f"F0 中央値 = {f0:.1f} Hz"
    return Classification(label, confidence, detail)


def classify_gender(features: AudioFeatures) -> Classification:
    """F0 中央値から性別を推定.

    - 一般に成人男性の平均 F0 は約 85〜180 Hz, 成人女性は約 165〜255 Hz.
    - 境界付近 (150〜180Hz) は信頼度を低くする.
    """
    f0 = features.f0_median_hz
    if f0 <= 0:
        return Classification("unknown", 0.0, "有声区間が検出できませんでした")

    # 子供の声域は男女ほぼ同等に高いため、非常に高い場合は判定保留
    if f0 > 260:
        return Classification(
            "unknown",
            0.4,
            f"F0 が {f0:.1f} Hz と高く、子供または女性いずれの可能性もあります",
        )

    boundary = 165.0  # 男女の一般的な境界値 (Hz)
    label: Gender = "female" if f0 >= boundary else "male"

    # 境界から離れるほど自信が高い (±40Hz で 0.5, ±80Hz 以上で 1.0)
    dist = abs(f0 - boundary)
    confidence = min(1.0, 0.5 + dist / 80.0)
    detail = f"F0 中央値 = {f0:.1f} Hz (境界 {boundary:.0f} Hz)"
    return Classification(label, confidence, detail)


def classify_age_group(features: AudioFeatures) -> Classification:
    """F0 とスペクトル特徴から年齢層を推定.

    非常に大雑把な目安:
      - 子供: F0 が 250Hz 以上
      - 若年成人: 声のエネルギー/抑揚が大きく、スペクトル重心が高め
      - 成人: 中庸
      - 高齢者: F0 のばらつきが大きくジッター傾向、スペクトル重心が下がる
    """
    f0 = features.f0_median_hz
    f0_std = features.f0_std_hz
    centroid = features.spectral_centroid_mean

    if f0 <= 0:
        return Classification("unknown", 0.0, "有声区間が検出できませんでした")

    # 子供判定
    if f0 >= 260:
        return Classification(
            "child",
            0.7,
            f"F0 中央値 {f0:.1f} Hz が子供の声域に該当",
        )

    # 高齢者の指標: F0 標準偏差が相対的に大きく、スペクトル重心が下がる
    #  - 若い人でも抑揚が大きいと大きくなるので、あくまで補助的.
    if f0_std > 45 and centroid < 1800:
        return Classification(
            "senior",
            0.55,
            f"F0 変動 {f0_std:.1f} Hz, 重心 {centroid:.0f} Hz (ジッター傾向)",
        )

    # 若年成人 vs 成人: スペクトル重心が高めで抑揚も大きい → 若年成人
    if centroid >= 2200 and f0_std >= 25:
        return Classification(
            "young_adult",
            0.6,
            f"明るめの声質 (重心 {centroid:.0f} Hz, F0変動 {f0_std:.1f} Hz)",
        )

    return Classification(
        "adult",
        0.55,
        f"標準的な声質 (重心 {centroid:.0f} Hz, F0変動 {f0_std:.1f} Hz)",
    )


def classify_voice_type(features: AudioFeatures, gender: Gender) -> Classification:
    """声楽的な声種 (バス〜ソプラノ) を F0 範囲から簡易推定.

    声楽分類は本来、歌唱音域や声区に基づくが、話者の F0 分布から目安をつける.
    """
    if features.f0_median_hz <= 0:
        return Classification("unknown", 0.0, "有声区間が検出できませんでした")

    f0_median = features.f0_median_hz

    # 声楽の声種境界 (Hz, 中央値ベースの大まかな目安)
    if gender == "male" or (gender == "unknown" and f0_median < 165):
        if f0_median < 110:
            label: VoiceType = "bass"
        elif f0_median < 140:
            label = "baritone"
        elif f0_median < 180:
            label = "tenor"
        else:
            label = "countertenor"
    else:  # female or unknown-but-high
        if f0_median < 200:
            label = "alto"
        elif f0_median < 260:
            label = "mezzo_soprano"
        else:
            label = "soprano"

    detail = f"F0 中央値 {f0_median:.1f} Hz, F0 幅 {features.f0_min_hz:.0f}-{features.f0_max_hz:.0f} Hz"
    return Classification(label, 0.5, detail)


def classify_emotion(features: AudioFeatures) -> Classification:
    """抑揚/音量/話速から感情を大まかに推定.

    精密な感情認識には音響+言語モデルが必要だが、ルールベースでも
    「落ち着いているか」「興奮しているか」の軸はある程度つかめる.
    """
    rms = features.rms_mean
    rms_std = features.rms_std
    f0_std = features.f0_std_hz
    onset_rate = features.onset_rate_per_sec
    centroid = features.spectral_centroid_mean

    # 正規化用の経験的閾値
    high_energy = rms > 0.05
    low_energy = rms < 0.015
    high_variation = f0_std > 40 or rms_std > 0.03
    fast_speech = onset_rate > 3.5
    slow_speech = onset_rate < 1.5
    bright_timbre = centroid > 2200

    if high_energy and high_variation and fast_speech:
        if bright_timbre:
            return Classification(
                "happy",
                0.55,
                f"高エネルギー・抑揚大・明るい声質 (RMS {rms:.3f}, F0σ {f0_std:.1f})",
            )
        return Classification(
            "angry",
            0.5,
            f"高エネルギー・抑揚大・硬い声質 (RMS {rms:.3f}, F0σ {f0_std:.1f})",
        )

    if high_energy and high_variation:
        return Classification(
            "excited",
            0.55,
            f"高エネルギー・抑揚あり (RMS {rms:.3f}, F0σ {f0_std:.1f})",
        )

    if low_energy and slow_speech and f0_std < 20:
        return Classification(
            "sad",
            0.55,
            f"低エネルギー・抑揚少・話速遅 (RMS {rms:.3f}, F0σ {f0_std:.1f})",
        )

    if not high_energy and not high_variation:
        return Classification(
            "calm",
            0.6,
            f"安定した発話 (RMS {rms:.3f}, F0σ {f0_std:.1f})",
        )

    return Classification(
        "neutral",
        0.5,
        f"特徴的な傾向なし (RMS {rms:.3f}, F0σ {f0_std:.1f}, onset {onset_rate:.1f}/s)",
    )
