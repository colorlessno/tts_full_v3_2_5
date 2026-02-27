#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_style_map.py

style_log.jsonl（voice_clone_full_pipeline.py の --log-style）から
「日本語キーワード → (pitch/energy/speed) の平均」を作り、style_map.json を吐く簡易学習。

使い方:
  python train_style_map.py --logs style_log.jsonl --out style_map.json

ログ1行の例（JSONL）:
{"time":"...","style_prompt":"落ち着いて...","mode":"nl","resolved":{"pitch":0.95,"energy":0.9,"speed":1.1}, ...}

出力（JSON）:
{
  "落ち着": {"pitch":0.95, "energy":0.9, "speed":1.1},
  ...
}

注意:
- これは最小実装。精度を上げたいなら「形態素解析」や「対話タグ付け」等を後から足せる。
- 乗算系の値なので、平均は log 空間（幾何平均）で取る。
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


JP_TOKEN_RE = re.compile(r"[一-龥ぁ-ゖァ-ヺー]{2,}")  # 日本語っぽい連続2文字以上


def extract_keywords(style_prompt: str):
    # 句読点/空白で軽く割る → 日本語連続トークン抽出
    chunks = re.split(r"[、,。．\s/]+", style_prompt)
    keys = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        for m in JP_TOKEN_RE.finditer(c):
            tok = m.group(0)
            # ありがちなノイズを落とす（最低限）
            if tok in ("ちょっと", "すこし", "少し", "感じ", "っぽい", "みたい"):
                continue
            keys.append(tok)
    # 重複削除（順序保持）
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def geom_mean(vals):
    # vals は正数前提
    logs = [math.log(max(v, 1e-6)) for v in vals]
    return float(math.exp(sum(logs) / max(1, len(logs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="style_log.jsonl path")
    ap.add_argument("--out", required=True, help="style_map.json output")
    ap.add_argument("--min-count", type=int, default=3, help="最低出現回数")
    ap.add_argument("--max-keys", type=int, default=200, help="最大キー数（多すぎ防止）")
    args = ap.parse_args()

    buckets = defaultdict(lambda: {"pitch": [], "energy": [], "speed": [], "count": 0})

    for obj in iter_jsonl(Path(args.logs)):
        prompt = str(obj.get("style_prompt", ""))
        resolved = obj.get("resolved") or {}
        if not all(k in resolved for k in ("pitch", "energy", "speed")):
            continue

        keys = extract_keywords(prompt)
        if not keys:
            continue

        p = float(resolved["pitch"])
        e = float(resolved["energy"])
        s = float(resolved["speed"])

        for k in keys:
            b = buckets[k]
            b["pitch"].append(p)
            b["energy"].append(e)
            b["speed"].append(s)
            b["count"] += 1

    # フィルタ & まとめ
    items = []
    for k, v in buckets.items():
        if v["count"] < args.min_count:
            continue
        items.append((k, v["count"], {
            "pitch": geom_mean(v["pitch"]),
            "energy": geom_mean(v["energy"]),
            "speed": geom_mean(v["speed"]),
        }))

    # 多すぎる時は count 上位
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:args.max_keys]

    out = {k: params for (k, _cnt, params) in items}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", args.out, "keys:", len(out))


if __name__ == "__main__":
    main()
