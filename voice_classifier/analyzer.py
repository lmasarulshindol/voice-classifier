"""特徴量抽出と各分類器を統合するアナライザ."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

from .classifiers import (
    Classification,
    classify_age_group,
    classify_emotion,
    classify_gender,
    classify_pitch,
    classify_voice_type,
)
from .features import AudioFeatures, extract_features


# 日本語の表示ラベル (CLI 出力用)
JP_LABELS: Dict[str, str] = {
    # pitch
    "very_low": "非常に低い",
    "low": "低い",
    "mid": "中くらい",
    "high": "高い",
    "very_high": "非常に高い",
    # gender
    "male": "男性",
    "female": "女性",
    # age
    "child": "子供",
    "young_adult": "若年成人",
    "adult": "成人",
    "senior": "高齢者",
    # voice type
    "bass": "バス",
    "baritone": "バリトン",
    "tenor": "テノール",
    "countertenor": "カウンターテナー",
    "alto": "アルト",
    "mezzo_soprano": "メゾソプラノ",
    "soprano": "ソプラノ",
    # emotion
    "calm": "落ち着き",
    "happy": "嬉しい",
    "excited": "興奮",
    "angry": "怒り",
    "sad": "悲しみ",
    "neutral": "中立",
    # fallback
    "unknown": "判定不能",
}


@dataclass(frozen=True)
class VoiceAnalysisResult:
    """声質分析の結果まとめ."""

    features: AudioFeatures
    pitch: Classification
    gender: Classification
    age_group: Classification
    voice_type: Classification
    emotion: Classification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": asdict(self.features),
            "pitch": asdict(self.pitch),
            "gender": asdict(self.gender),
            "age_group": asdict(self.age_group),
            "voice_type": asdict(self.voice_type),
            "emotion": asdict(self.emotion),
        }


class VoiceAnalyzer:
    """音声ファイルを受け取り、5 種類の分類を実施する."""

    def analyze(self, audio_path: Union[str, Path]) -> VoiceAnalysisResult:
        features = extract_features(audio_path)

        pitch = classify_pitch(features)
        gender = classify_gender(features)
        age_group = classify_age_group(features)
        # voice_type は gender の結果に依存
        voice_type = classify_voice_type(features, gender.label)  # type: ignore[arg-type]
        emotion = classify_emotion(features)

        return VoiceAnalysisResult(
            features=features,
            pitch=pitch,
            gender=gender,
            age_group=age_group,
            voice_type=voice_type,
            emotion=emotion,
        )


def format_result(result: VoiceAnalysisResult, use_japanese: bool = True) -> str:
    """結果を人間が読みやすい文字列に整形する."""

    def label(value: str) -> str:
        return JP_LABELS.get(value, value) if use_japanese else value

    f = result.features

    lines = [
        "=" * 50,
        "  音声分析結果",
        "=" * 50,
        f"  長さ            : {f.duration_sec:.2f} 秒",
        f"  サンプリング    : {f.sample_rate} Hz",
        f"  F0 中央値       : {f.f0_median_hz:.1f} Hz",
        f"  F0 範囲         : {f.f0_min_hz:.1f} - {f.f0_max_hz:.1f} Hz",
        f"  F0 標準偏差     : {f.f0_std_hz:.1f} Hz",
        f"  有声区間比率    : {f.voiced_ratio * 100:.1f} %",
        f"  平均音量 (RMS)  : {f.rms_mean:.4f}",
        f"  スペクトル重心  : {f.spectral_centroid_mean:.1f} Hz",
        f"  オンセット密度  : {f.onset_rate_per_sec:.2f} /秒",
        "-" * 50,
        "  分類結果",
        "-" * 50,
    ]

    def render(name: str, c: Classification) -> str:
        conf_bar = "*" * int(round(c.confidence * 10))
        return (
            f"  {name:<10}: {label(c.label):<12} "
            f"信頼度 {c.confidence:.2f} [{conf_bar:<10}]\n"
            f"             └ {c.detail}"
        )

    lines.append(render("声の高さ", result.pitch))
    lines.append(render("性別", result.gender))
    lines.append(render("年齢層", result.age_group))
    lines.append(render("声種", result.voice_type))
    lines.append(render("感情傾向", result.emotion))
    lines.append("=" * 50)

    return "\n".join(lines)
