#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phoneme_align.py (template)

- v3 の "音素区間まで含めて強化" は voice_clone_full_pipeline.py 側で
  G2P/文字を等分割して区間化することで実装しています。
- ここは forced alignment 実装に差し替えるときの土台。

依存（任意）:
  pip install pyopenjtalk soundfile
  ffmpeg が PATH に必要
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf


def ffmpeg_to_wav(src: Path, dst: Path, sr=16000):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    subprocess.check_call([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr),
        str(dst)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def g2p_pyopenjtalk(text: str):
    try:
        import pyopenjtalk  # type: ignore
    except Exception:
        return None
    ph = pyopenjtalk.g2p(text, kana=False)
    toks = [p for p in ph.split() if p.strip()]
    return toks if toks else None


def get_duration_sec(wav_path: Path):
    y, sr = sf.read(str(wav_path))
    if hasattr(y, "ndim") and y.ndim > 1:
        y = y[:, 0]
    return float(len(y) / sr), int(sr)


def uniform_align(units, duration_sec: float):
    n = max(1, len(units))
    step = duration_sec / n if duration_sec > 0 else 0.0
    out = []
    t = 0.0
    for u in units:
        out.append({"p": u, "start": t, "end": t + step})
        t += step
    out[-1]["end"] = duration_sec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", help="if omitted, use --text-file")
    ap.add_argument("--text-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    if args.text is None and args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "in.wav"
        ffmpeg_to_wav(Path(args.audio), wav, sr=args.sr)
        duration_sec, sr = get_duration_sec(wav)

        units = g2p_pyopenjtalk(text)
        if units is None:
            units = [c for c in text if c.strip()]

        segments = uniform_align(units, duration_sec)

        out = {
            "audio": str(Path(args.audio)),
            "sr": sr,
            "duration_sec": duration_sec,
            "text": text,
            "units": units,
            "segments": segments,
            "notes": "uniform template; replace with forced alignment if needed",
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
