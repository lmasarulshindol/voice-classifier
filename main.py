"""音声ファイルを受け取って声のタイプを分類する CLI.

使い方:
    python main.py path/to/audio.wav
    python main.py path/to/audio.mp3 --json
    python main.py *.wav            # 複数ファイルまとめて分析
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voice_classifier.analyzer import VoiceAnalyzer, format_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="音声ファイルから声のタイプ (高さ/性別/年齢層/声種/感情) を分類する",
    )
    parser.add_argument(
        "audio_paths",
        nargs="+",
        help="分析する音声ファイル (wav/mp3/m4a/flac など)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果を JSON 形式で出力",
    )
    parser.add_argument(
        "--english",
        action="store_true",
        help="ラベルを英語で出力 (デフォルトは日本語)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    analyzer = VoiceAnalyzer()
    all_results = []
    had_error = False

    for raw_path in args.audio_paths:
        path = Path(raw_path)
        try:
            result = analyzer.analyze(path)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            had_error = True
            continue
        except Exception as exc:  # noqa: BLE001 - ユーザーへの表示用
            print(f"[ERROR] {path} の解析に失敗しました: {exc}", file=sys.stderr)
            had_error = True
            continue

        if args.json:
            all_results.append({"file": str(path), **result.to_dict()})
        else:
            print(f"\n[ファイル] {path}")
            print(format_result(result, use_japanese=not args.english))

    if args.json:
        json.dump(all_results, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
