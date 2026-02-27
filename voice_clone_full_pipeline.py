#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
voice_clone_full_pipeline.py (v3)
--------------------------------
- Whisper 実装を直で組み込む（ref_audio → ref_text 自動生成）
- qwen3-tts の pitch/energy フックは「後処理（pyworld）」で確実に接続
- さらに "音素区間（=phoneme segments）" まで含めて強化:
    - 生成テキストから G2P (pyopenjtalkがあれば) → phoneme 列
    - phoneme 数 N に合わせて prosody-ref（抑揚参照音声）を N 分割
    - N 区間ごとの平均F0 / 平均Energy を抽出し、生成音声の N 区間へ反映（境界平滑化）

依存（推奨）:
  pip install numpy soundfile torch librosa pyworld
  (Whisper) pip install faster-whisper
  (G2P)    pip install pyopenjtalk
  ffmpeg が PATH に必要
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import librosa
import pyworld as pw


# =========================
# Utils
# =========================

def ffmpeg_to_wav(src: Path, dst: Path, sr: Optional[int] = 16000, debug: bool = False):
    """
    ffmpeg で src をモノラル wav(指定sr) に変換して dst に保存。
    失敗時は stderr を表示する（原因が見えるようにする）。
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn", "-ac", "1",
    ]
    if sr is not None:
        cmd += ["-ar", str(sr)]
    cmd += [str(dst)]

    if debug:
        r = subprocess.run(cmd)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        stderr = r.stderr if getattr(r, "stderr", None) else ""
        raise RuntimeError(
            "ffmpeg failed\n"
            f"  src: {src}\n"
            f"  dst: {dst}\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  returncode: {r.returncode}\n"
            f"  stderr:\n{stderr}"
        )


def ensure_wav_for_option(src: Path, temp_dir: Path, tag: str, sr: Optional[int], debug: bool = False) -> Path:
    """
    オプションで渡された音声が wav でなければ、内部で wav へ変換して返す。
    - wav の場合: そのまま返す
    - mp3/mp4 等: ffmpeg で temp_dir 下へ変換
    """
    if not src.exists():
        raise RuntimeError(f"audio file not found: {src}")
    if src.suffix.lower() == ".wav":
        return src

    safe_stem = re.sub(r"[^0-9A-Za-z_-]+", "_", src.stem).strip("_") or "audio"
    dst = temp_dir / f"{tag}_{safe_stem}.wav"
    n = 1
    while dst.exists():
        dst = temp_dir / f"{tag}_{safe_stem}_{n}.wav"
        n += 1

    ffmpeg_to_wav(src, dst, sr=sr, debug=debug)
    if debug:
        print(f"[auto-wav] {src} -> {dst}")
    return dst


def read_text_flexible(path: Path, label: str = "text") -> str:
    """
    UTF-8 以外（cp932 / shift_jis など）で保存されたテキストも読めるようにする。
    """
    encodings = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp"]
    last_err = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(
        f"failed to decode {label}: {path}\n"
        f"tried encodings: {', '.join(encodings)}\n"
        f"last error: {last_err}\n"
        "file encoding を UTF-8 か Shift_JIS(cp932) にしてください。"
    )


def read_lines(path: Path):
    return [l.strip() for l in read_text_flexible(path, label="text lines").splitlines() if l.strip()]


def concat_with_silence(wavs, sr, silence=0.25):
    sil = np.zeros(int(sr * silence), np.float32)
    out = []
    for i, w in enumerate(wavs):
        out.append(w.astype(np.float32))
        if i != len(wavs) - 1 and silence > 0:
            out.append(sil)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def resample_1d(x: np.ndarray, n: int):
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) == n:
        return x.astype(np.float32)
    xp = np.linspace(0, 1, num=len(x), dtype=np.float32)
    xq = np.linspace(0, 1, num=n, dtype=np.float32)
    return np.interp(xq, xp, x.astype(np.float32)).astype(np.float32)


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def load_json_object(path: Path, label: str) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to read {label} JSON: {path}\nerror: {e}")
    if not isinstance(obj, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return obj


def compile_pron_map(raw_map: dict) -> List[Tuple[str, str]]:
    rules: List[Tuple[str, str]] = []
    for src, dst in raw_map.items():
        s = str(src).strip()
        d = str(dst).strip()
        if not s or not d:
            continue
        rules.append((s, d))
    # 長いキーを先に適用して、短いキーへの部分一致で壊れないようにする
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules


def apply_pron_map(text: str, rules: List[Tuple[str, str]]) -> str:
    out = text
    for src, dst in rules:
        out = out.replace(src, dst)
    return out


def _norm_path_for_compare(path: Path) -> str:
    try:
        return str(path.expanduser().resolve(strict=False)).lower()
    except Exception:
        return str(path).lower()


def warn_prosody_settings(ref_audio: Path, prosody_paths: List[Path], strength: float, smooth: int):
    if not prosody_paths:
        return
    if strength > 0.5 and smooth < 7:
        print(
            "[warn] prosody が強すぎる可能性があります。"
            "--prosody-strength 0.1〜0.3 と --prosody-smooth 7〜11 を検討してください。",
            file=sys.stderr,
        )
    ref_norm = _norm_path_for_compare(ref_audio)
    same_ref = any(_norm_path_for_compare(p) == ref_norm for p in prosody_paths)
    if same_ref and strength >= 0.5:
        print(
            "[warn] --prosody-ref に --ref-audio と同じ音声が含まれています。"
            "高い prosody-strength は音色がこもる原因になりやすいです。",
            file=sys.stderr,
        )


def _quote_for_cmd(s: str) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'


def run_command_checked(cmd, debug: bool = False, cwd: Optional[Path] = None, shell: bool = False):
    if debug:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=shell)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None, shell=shell)
    if r.returncode != 0:
        stderr = r.stderr if getattr(r, "stderr", None) else ""
        stdout = r.stdout if getattr(r, "stdout", None) else ""
        raise RuntimeError(
            "command failed\n"
            f"  cmd: {cmd}\n"
            f"  returncode: {r.returncode}\n"
            f"  stdout:\n{stdout}\n"
            f"  stderr:\n{stderr}"
        )


def vc_convert_external_cmd(
    cmd_template: str,
    in_wav: Path,
    out_wav: Path,
    source_audio: Optional[Path],
    ref_audio: Optional[Path],
    vc_model: Optional[str],
    vc_index: Optional[str],
    debug: bool = False,
    workdir: Optional[Path] = None,
):
    if not cmd_template:
        raise RuntimeError("--vc-backend external_cmd には --vc-cmd-template が必要です")

    vals = {
        "in_wav": str(in_wav),
        "out_wav": str(out_wav),
        "source_audio": str(source_audio) if source_audio else "",
        "ref_audio": str(ref_audio) if ref_audio else "",
        "vc_model": str(vc_model) if vc_model else "",
        "vc_index": str(vc_index) if vc_index else "",
    }
    vals_q = {f"{k}_q": _quote_for_cmd(v) for k, v in vals.items()}
    fmt = {}
    fmt.update(vals)
    fmt.update(vals_q)

    try:
        cmd_text = cmd_template.format(**fmt)
    except KeyError as e:
        raise RuntimeError(
            f"vc-cmd-template placeholder error: missing {e}. "
            "use keys: in_wav/out_wav/source_audio/ref_audio/vc_model/vc_index (+ *_q)."
        )

    run_command_checked(cmd_text, debug=debug, cwd=workdir, shell=True)
    if not out_wav.exists() or out_wav.stat().st_size < 1024:
        raise RuntimeError(f"VC output not created or too small: {out_wav}")


def run_vc_backend(
    backend: str,
    in_wav: Path,
    out_wav: Path,
    source_audio: Optional[Path],
    ref_audio: Optional[Path],
    vc_cmd_template: Optional[str],
    vc_model: Optional[str],
    vc_index: Optional[str],
    vc_workdir: Optional[str],
    debug: bool = False,
):
    backend = (backend or "none").lower()
    if backend == "none":
        shutil.copyfile(str(in_wav), str(out_wav))
        return
    if backend == "external_cmd":
        wd = Path(vc_workdir) if vc_workdir else None
        vc_convert_external_cmd(
            cmd_template=vc_cmd_template or "",
            in_wav=in_wav,
            out_wav=out_wav,
            source_audio=source_audio,
            ref_audio=ref_audio,
            vc_model=vc_model,
            vc_index=vc_index,
            debug=debug,
            workdir=wd,
        )
        return
    raise RuntimeError(f"unknown vc backend: {backend}")


def _safe_repo_dir(repo_id: str) -> str:
    s = (repo_id or "").strip().replace("\\", "/")
    s = s.replace("/", "__").replace(":", "_")
    return s or "repo"


def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists() and p.is_file():
            return p
    return None


def _pick_best_file(candidates: List[Path], prefer_tokens: List[str]) -> Optional[Path]:
    if not candidates:
        return None

    def _score(p: Path):
        name = p.name.lower()
        score = 0
        for i, tok in enumerate(prefer_tokens):
            if tok in name:
                score += (len(prefer_tokens) - i)
        return (-score, len(str(p)))

    return sorted(candidates, key=_score)[0]


def setup_vc_assets_from_hf(
    repo_id: str,
    local_root: Path,
    revision: Optional[str] = None,
    subdir: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    explicit_model: Optional[str] = None,
    explicit_index: Optional[str] = None,
    force_download: bool = False,
    debug: bool = False,
) -> dict:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 連携には huggingface_hub が必要です。\n"
            "pip install huggingface_hub\n"
            f"import error: {e}"
        )

    local_root.mkdir(parents=True, exist_ok=True)
    repo_dir = local_root / _safe_repo_dir(repo_id)

    allow = allow_patterns if allow_patterns else None
    if debug:
        print(f"[vc-hf] downloading repo={repo_id} revision={revision or 'main'} -> {repo_dir}")
        if allow:
            print(f"[vc-hf] allow_patterns={allow}")

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(repo_dir),
        "revision": revision,
        "token": token,
        "allow_patterns": allow,
        "resume_download": True,
        "force_download": bool(force_download),
    }
    # revision=None をそのまま渡すと古いバージョンで警告になることがある
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        snapshot_download(**kwargs)
    except TypeError:
        # huggingface_hub の古い版向けフォールバック
        kwargs.pop("resume_download", None)
        kwargs.pop("force_download", None)
        snapshot_download(**kwargs)

    base_dir = (repo_dir / subdir) if subdir else repo_dir
    if not base_dir.exists():
        raise RuntimeError(f"vc-hf subdir not found: {base_dir}")

    model_candidates = sorted([
        p for p in base_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".pth", ".pt", ".onnx", ".bin")
    ])
    index_candidates = sorted([
        p for p in base_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".index"
    ])

    explicit_model_path = Path(explicit_model) if explicit_model else None
    explicit_index_path = Path(explicit_index) if explicit_index else None
    if explicit_model_path and not explicit_model_path.exists():
        candidate = base_dir / explicit_model_path
        explicit_model_path = candidate if candidate.exists() else explicit_model_path
    if explicit_index_path and not explicit_index_path.exists():
        candidate = base_dir / explicit_index_path
        explicit_index_path = candidate if candidate.exists() else explicit_index_path

    model_path = _pick_first_existing(
        [explicit_model_path] if explicit_model_path else []
    ) or _pick_best_file(model_candidates, ["model", "voice", "rvc", "g_", "generator"])
    index_path = _pick_first_existing(
        [explicit_index_path] if explicit_index_path else []
    ) or _pick_best_file(index_candidates, ["index", "added", "trained"])

    if model_path is None:
        raise RuntimeError(
            f"VC model file not found in downloaded repo: {base_dir}\n"
            "set --vc-model explicitly or adjust --vc-hf-allow-pattern."
        )

    manifest = {
        "repo_id": repo_id,
        "revision": revision or "main",
        "base_dir": str(base_dir),
        "model": str(model_path),
        "index": str(index_path) if index_path else None,
        "model_candidates": [str(p) for p in model_candidates[:20]],
        "index_candidates": [str(p) for p in index_candidates[:20]],
    }
    manifest_path = repo_dir / "vc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)

    if debug:
        print(f"[vc-hf] manifest={manifest_path}")
        print(f"[vc-hf] model={manifest['model']}")
        if manifest["index"]:
            print(f"[vc-hf] index={manifest['index']}")
    return manifest


# =========================
# Style-prompt handling
# =========================

def is_numeric_dsl(prompt: str):
    return ":" in prompt and "%" in prompt


def parse_numeric_dsl(prompt: str):
    params = {"pitch": 1.0, "energy": 1.0, "speed": 1.0}
    for tok in prompt.replace(",", " ").split():
        if ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        if k not in params:
            continue
        v = v.strip()
        try:
            if v.endswith("%"):
                params[k] *= 1.0 + float(v[:-1]) / 100.0
            else:
                params[k] *= float(v)
        except ValueError:
            continue
    return params


def nl_prompt_to_params(prompt: str, style_map: dict):
    params = {"pitch": 1.0, "energy": 1.0, "speed": 1.0}
    p = prompt.lower()
    for key in sorted(style_map.keys(), key=len, reverse=True):
        if key.lower() in p:
            val = style_map[key]
            for k in ("pitch", "energy", "speed"):
                if k in val:
                    params[k] *= float(val[k])
    return params


DEFAULT_STYLE_MAP = {
    "落ち着": {"pitch": 0.95, "energy": 0.9, "speed": 1.1},
    "苛立": {"pitch": 1.05, "energy": 1.1, "speed": 0.95},
    "怒": {"pitch": 1.15, "energy": 1.3, "speed": 0.9},
    "皮肉": {"pitch": 0.98, "energy": 1.0, "speed": 1.05},
    "意味わから": {"pitch": 1.02, "energy": 1.0, "speed": 1.0},
}

STYLE_PARAM_LIMITS = {
    "pitch": (0.7, 1.3),
    "energy": (0.6, 1.6),
    "speed": (0.8, 1.2),
}


def clamp_style_params(params: dict) -> dict:
    out = {}
    for k, (lo, hi) in STYLE_PARAM_LIMITS.items():
        try:
            v = float(params.get(k, 1.0))
        except Exception:
            v = 1.0
        out[k] = float(np.clip(v, lo, hi))
    return out


def extract_first_json_object(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        raise RuntimeError("LLM response is empty")

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL)
    if m:
        t = m.group(1).strip()

    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(t[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError(f"LLM response does not contain JSON object: {t[:200]}")


def call_openai_compatible_chat(
    endpoint: str,
    model: str,
    messages: list,
    api_key: Optional[str],
    timeout_sec: float,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"LLM HTTPError {e.code}: {detail}")
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")

    try:
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM response parse failed: {e}\nraw={raw[:400]}")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Responses API互換の text part をつなぐ
        parts = []
        for c in content:
            if isinstance(c, dict):
                if "text" in c:
                    parts.append(str(c["text"]))
                elif c.get("type") == "output_text" and "text" in c:
                    parts.append(str(c["text"]))
        return "".join(parts).strip()
    return str(content)


def llm_prompt_to_params(
    prompt: str,
    style_map: dict,
    endpoint: str,
    model: str,
    api_key: Optional[str],
    timeout_sec: float,
    temperature: float,
    max_tokens: int,
    feedback: str = "",
    debug_file: Optional[str] = None,
) -> dict:
    system_prompt = (
        "You convert Japanese TTS style prompts into numeric prosody multipliers.\n"
        "Return JSON only with keys: pitch, energy, speed.\n"
        "Target ranges: pitch 0.7-1.3, energy 0.6-1.6, speed 0.8-1.2."
    )
    user_prompt = (
        "style_prompt:\n"
        f"{prompt}\n\n"
        "current_style_map (hint only):\n"
        f"{json.dumps(style_map, ensure_ascii=False)}\n\n"
        "feedback (optional):\n"
        f"{feedback or '(none)'}\n\n"
        "Output JSON example:\n"
        "{\"pitch\": 1.03, \"energy\": 0.95, \"speed\": 1.02}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = call_openai_compatible_chat(
        endpoint=endpoint,
        model=model,
        messages=messages,
        api_key=api_key,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    obj = extract_first_json_object(raw)
    params = clamp_style_params(obj)

    if debug_file:
        d = Path(debug_file)
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(json.dumps({
            "endpoint": endpoint,
            "model": model,
            "style_prompt": prompt,
            "feedback": feedback,
            "messages": messages,
            "raw_response": raw,
            "parsed": obj,
            "clamped": params,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return params


def merge_style_params(rule_params: dict, llm_params: dict) -> dict:
    # ルール値とLLM値の中間を幾何平均で取って暴れを抑える
    out = {}
    for k in ("pitch", "energy", "speed"):
        a = float(rule_params.get(k, 1.0))
        b = float(llm_params.get(k, 1.0))
        out[k] = math.sqrt(max(a, 1e-6) * max(b, 1e-6))
    return clamp_style_params(out)


# =========================
# Whisper (ref_text auto)
# =========================

def whisper_transcribe(
    wav_path: Path,
    language: str = "ja",
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 1,
    debug: bool = False,
) -> str:
    """
    faster-whisper が入っていればそれを使用。
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Whisper 直組みには faster-whisper が必要です。\n"
            "pip install faster-whisper\n"
            f"import error: {e}"
        )
    actual_device = device
    if actual_device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_compute_type = compute_type
    if actual_compute_type == "auto":
        actual_compute_type = "float16" if actual_device == "cuda" else "int8"

    if debug:
        print(f"[whisper] model={model_size} device={actual_device} compute_type={actual_compute_type} language={language}")

    try:
        model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute_type)
        segments, info = model.transcribe(str(wav_path), language=language, beam_size=int(beam_size))
    except Exception as e:
        # auto+cuda が不安定な環境向けに CPU へフォールバック
        if device == "auto" and actual_device == "cuda":
            if debug:
                print(f"[warn] whisper cuda failed, retry on cpu: {e}")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(wav_path), language=language, beam_size=int(beam_size))
        else:
            raise
    text = "".join([s.text for s in segments]).strip()
    return text


# =========================
# Prosody analysis (F0 / Energy)
# =========================

def extract_f0(y: np.ndarray, sr: int):
    f0, t = pw.harvest(y.astype(np.float64), sr)
    f0 = f0.astype(np.float32)
    f0[f0 < 50] = 0
    return f0, t


def parse_weighted_paths(items):
    """
    path[:weight] を複数受け取る。
    Windows の絶対パス (例: C:\\path\\file.wav) を壊さないように、
    末尾の ":<float>" だけを weight として解釈する。
    """
    paths = []
    weights = []
    for item in items:
        p = item
        w = 1.0
        if ":" in item:
            p2, tail = item.rsplit(":", 1)
            try:
                w = float(tail)
                p = p2
            except ValueError:
                p = item
                w = 1.0
        paths.append(Path(p))
        weights.append(float(w))
    return paths, weights


def blend_profiles(profiles: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if not profiles:
        return np.zeros(0, np.float32)
    n = min(len(p) for p in profiles)
    stack = np.stack([p[:n] for p in profiles]).astype(np.float32)
    w = np.array(weights, np.float32).reshape(-1, 1)
    denom = float(w.sum()) if float(w.sum()) != 0 else 1.0
    return (stack * w).sum(axis=0) / denom


# =========================
# G2P & segments
# =========================

def g2p(text: str) -> Optional[List[str]]:
    try:
        import pyopenjtalk  # type: ignore
    except Exception:
        return None
    ph = pyopenjtalk.g2p(text, kana=False)
    toks = [p for p in ph.split() if p.strip()]
    return toks if toks else None


def fallback_units(text: str) -> List[str]:
    return [c for c in text if c.strip()]


def make_uniform_segments(n: int, duration_sec: float) -> List[tuple]:
    n = max(1, int(n))
    step = duration_sec / n if duration_sec > 0 else 0.0
    segs = []
    t = 0.0
    for _ in range(n):
        segs.append((t, min(duration_sec, t + step)))
        t += step
    if segs:
        segs[-1] = (segs[-1][0], duration_sec)
    return segs


def make_phoneme_segments(text: str, duration_sec: float, override_n: int = 0):
    if override_n and override_n > 0:
        units = [f"seg{i}" for i in range(override_n)]
    else:
        units = g2p(text) or fallback_units(text)
    segs = make_uniform_segments(len(units), duration_sec)
    return units, segs


def segment_f0_means(f0: np.ndarray, t: np.ndarray, segments: List[tuple]) -> np.ndarray:
    out = np.zeros(len(segments), np.float32)
    for i, (a, b) in enumerate(segments):
        mask = (t >= a) & (t < b)
        v = f0[mask]
        v = v[v > 0]
        out[i] = float(v.mean()) if v.size > 0 else 0.0
    return out


def segment_rms_means(y: np.ndarray, sr: int, segments: List[tuple]) -> np.ndarray:
    out = np.zeros(len(segments), np.float32)
    for i, (a, b) in enumerate(segments):
        ia = int(max(0, round(a * sr)))
        ib = int(min(len(y), round(b * sr)))
        seg = y[ia:ib]
        if seg.size == 0:
            out[i] = 0.0
        else:
            out[i] = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2)))
    return out


def smooth_step_profile(vals: np.ndarray, smooth: int) -> np.ndarray:
    if smooth <= 1:
        return vals.astype(np.float32)
    kernel = np.ones(smooth, np.float32) / float(smooth)
    pad = smooth // 2
    v = np.pad(vals.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(v, kernel, mode="valid").astype(np.float32)


def build_f0_multiplier(src_f0: np.ndarray, t: np.ndarray, segs: List[tuple], tgt_f0_means: np.ndarray,
                        strength: float, style_pitch: float, smooth_segments: int) -> np.ndarray:
    n = min(len(segs), len(tgt_f0_means))
    ratios = np.ones(n, np.float32)
    for i in range(n):
        a, b = segs[i]
        mask = (t >= a) & (t < b)
        v = src_f0[mask]
        v = v[v > 0]
        src_mean = float(v.mean()) if v.size > 0 else 0.0
        tgt_mean = float(tgt_f0_means[i])
        if src_mean <= 0 or tgt_mean <= 0:
            ratios[i] = 1.0
        else:
            ratios[i] = float(np.clip(tgt_mean / src_mean, 0.5, 2.0))
    ratios = 1.0 + (ratios - 1.0) * clamp01(strength)
    ratios = ratios * float(style_pitch)
    ratios = smooth_step_profile(ratios, smooth_segments)

    mult = np.ones_like(src_f0, np.float32)
    for i in range(n):
        a, b = segs[i]
        mask = (t >= a) & (t < b)
        mult[mask] = float(ratios[i])
    return mult


def apply_energy_by_segments(y: np.ndarray, sr: int, segs: List[tuple], tgt_rms: np.ndarray,
                             strength: float, style_energy: float, ramp_ms: float = 10.0) -> np.ndarray:
    y = y.astype(np.float32).copy()
    n = min(len(segs), len(tgt_rms))
    src_rms = segment_rms_means(y, sr, segs)[:n].astype(np.float32)

    gains = np.ones(n, np.float32)
    eps = 1e-6
    for i in range(n):
        if src_rms[i] <= 0 or tgt_rms[i] <= 0:
            gains[i] = 1.0
        else:
            gains[i] = float(np.clip(tgt_rms[i] / (src_rms[i] + eps), 0.5, 2.0))
    gains = 1.0 + (gains - 1.0) * clamp01(strength)
    gains = gains * float(style_energy)

    ramp = int(sr * (ramp_ms / 1000.0))
    ramp = max(1, ramp)

    for i in range(n):
        a, b = segs[i]
        ia = int(max(0, round(a * sr)))
        ib = int(min(len(y), round(b * sr)))
        if ib <= ia:
            continue
        g = float(gains[i])
        seg = y[ia:ib]
        if seg.size <= 2 * ramp:
            y[ia:ib] = seg * g
            continue
        y[ia:ia+ramp] = seg[:ramp] * np.linspace(1.0, g, ramp, dtype=np.float32)
        y[ia+ramp:ib-ramp] = seg[ramp:-ramp] * g
        y[ib-ramp:ib] = seg[-ramp:] * np.linspace(g, 1.0, ramp, dtype=np.float32)

    return np.clip(y, -1.0, 1.0)


def apply_pitch_energy_segmentwise(
    y: np.ndarray,
    sr: int,
    text: str,
    tgt_f0_means: Optional[np.ndarray],
    tgt_rms_means: Optional[np.ndarray],
    strength: float,
    style_pitch: float,
    style_energy: float,
    smooth_segments: int,
    override_segments: int = 0,
    do_world_pitch: bool = True,
) -> np.ndarray:
    duration_sec = float(len(y) / sr)
    _, segs = make_phoneme_segments(text, duration_sec, override_n=override_segments)

    y64 = y.astype(np.float64)
    peak = float(np.max(np.abs(y64))) if y64.size else 0.0
    if peak > 1.2:
        y64 = y64 / peak
        y = (y.astype(np.float32) / peak).astype(np.float32)

    # start from original
    y2 = y.astype(np.float32)

    # Pitch (WORLD synth) - optional
    if do_world_pitch:
        f0, t = extract_f0(y.astype(np.float32), sr)
        voiced = f0 > 0
    else:
        f0 = None
        t = None
        voiced = None

    if do_world_pitch and f0 is not None and t is not None and voiced is not None and voiced.sum() > 5:
        if tgt_f0_means is not None and len(tgt_f0_means) > 0:
            mult = build_f0_multiplier(f0, t, segs, tgt_f0_means, strength, style_pitch, smooth_segments)
            new_f0 = f0.copy()
            new_f0[voiced] = new_f0[voiced] * mult[voiced]
        else:
            new_f0 = f0.copy()
            new_f0[voiced] = new_f0[voiced] * float(style_pitch)

        # IMPORTANT: analyze with original f0, synthesize with modified f0
        sp = pw.cheaptrick(y64, f0.astype(np.float64), t, sr)
        ap = pw.d4c(y64, f0.astype(np.float64), t, sr)
        y2 = pw.synthesize(new_f0.astype(np.float64), sp, ap, sr).astype(np.float32)

    # Energy (segmentwise gain) - always possible
    if tgt_rms_means is not None and len(tgt_rms_means) > 0:
        y2 = apply_energy_by_segments(y2, sr, segs, tgt_rms_means, strength, style_energy, ramp_ms=10.0)
    else:
        y2 = np.clip(y2 * float(style_energy), -1.0, 1.0).astype(np.float32)

    return y2.astype(np.float32)


# =========================
# Speed

# =========================

def _ffmpeg_atempo_chain(rate: float) -> str:
    """
    ffmpeg atempo は 0.5〜2.0 の範囲制約があるので、範囲内に収まるように分解する。
    例: 3.0 -> atempo=2.0,atempo=1.5
        0.25 -> atempo=0.5,atempo=0.5
    """
    if rate <= 0:
        return "atempo=1.0"
    parts = []
    r = float(rate)
    while r > 2.0:
        parts.append(2.0)
        r /= 2.0
    while r < 0.5:
        parts.append(0.5)
        r /= 0.5
    parts.append(r)
    return ",".join([f"atempo={p:.6f}" for p in parts])


def apply_speed(y: np.ndarray, sr: int, rate: float, method: str = "ffmpeg"):
    """
    速度変更（time-stretch）
    - ffmpeg(atempo) は一般に librosa(phase vocoder) より水っぽさが出にくい
    """
    if rate <= 0:
        return y
    if abs(rate - 1.0) < 1e-6:
        return y

    if method == "ffmpeg" and shutil.which("ffmpeg") is not None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inp = td / "in.wav"
            outp = td / "out.wav"
            sf.write(str(inp), y.astype(np.float32), sr)
            flt = _ffmpeg_atempo_chain(rate)
            cmd = ["ffmpeg", "-y", "-i", str(inp), "-filter:a", flt, str(outp)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and outp.exists():
                y2, _sr = sf.read(str(outp))
                if hasattr(y2, "ndim") and y2.ndim > 1:
                    y2 = y2[:, 0]
                return np.asarray(y2, dtype=np.float32)
            # ffmpeg失敗時は librosaへフォールバック

    # librosa fallback（phase vocoder）
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate).astype(np.float32)


def load_qwen3_tts(model_id: str, device: str, dtype: str, attn: str):
    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "qwen3-tts 呼び出し部が未接続です。\n"
            "あなたの環境の TTS 実装に合わせて load_qwen3_tts()/generate_voice_clone() を差し替えてください。\n"
            f"import error: {e}"
        )

    torch_dtype = {
        "auto": (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype.lower(), torch.float16)

    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch_dtype,
        attn_implementation=attn,
    )


def generate_voice_clone(model, text: str, language: str, ref_audio_wav: str, ref_text: str):
    audio_list, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio_wav,
        ref_text=ref_text,
    )
    y = audio_list[0]
    return np.asarray(y, dtype=np.float32), int(sr)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ref-audio",
                    help="声質の参照音声（TTS経由時に必須）")
    ap.add_argument("--ref-text-file")
    ap.add_argument("--whisper-ref-text", action="store_true")
    ap.add_argument("--ref-text-out")

    ap.add_argument("--text-file", required=False,
                    help="合成するテキスト。未指定時は --source-audio 由来テキストを使う")
    ap.add_argument("--source-audio",
                    help="内容/抑揚の元音声。--text-file 未指定時はWhisperで文字起こしして本文に使う")
    ap.add_argument("--source-text-file",
                    help="source-audio の文字起こし済みテキスト（--text-file より優先しない）")
    ap.add_argument("--source-text-out",
                    help="source-audio をWhisper起こししたテキストの保存先")
    ap.add_argument("--source-whisper-language", default="ja",
                    help="source-audio 文字起こし時のWhisper language（例: ja, en）")
    ap.add_argument("--whisper-model", default="small",
                    help="Whisper model size/name（例: tiny, base, small）")
    ap.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Whisper 実行デバイス")
    ap.add_argument("--whisper-compute-type", default="auto",
                    help="Whisper compute_type（auto/int8/float16/float32 等）")
    ap.add_argument("--whisper-beam-size", type=int, default=1,
                    help="Whisper beam size")
    ap.add_argument("--source-as-prosody-ref", dest="source_as_prosody_ref", action="store_true", default=True,
                    help="source-audio を prosody-ref として自動利用する（デフォルト: ON）")
    ap.add_argument("--no-source-as-prosody-ref", dest="source_as_prosody_ref", action="store_false",
                    help="source-audio の prosody-ref 自動利用を無効化する")
    ap.add_argument("--no-auto-prosody-by-phoneme", action="store_true",
                    help="source-audio 利用時の prosody-by-phoneme 自動ONを無効化する")

    ap.add_argument("--prosody-ref", action="append", default=[])
    ap.add_argument("--prosody-by-phoneme", action="store_true")
    ap.add_argument("--prosody-segments", type=int, default=0)
    ap.add_argument("--prosody-strength", type=float, default=1.0)
    ap.add_argument("--prosody-smooth", type=int, default=3)

    ap.add_argument("--no-world-pitch", action="store_true",
                    help="WORLDでのpitch変更を無効化（ノイズ回避の緊急用）")

    ap.add_argument("--no-speed", action="store_true",
                    help="速度変更（time-stretch）を無効化（泥水/かぼす音回避）")
    ap.add_argument("--speed-override", type=float, default=None,
                    help="speed を強制指定（例: 1.0 / 0.95 / 1.05）。指定すると style-prompt の speed を無視")
    ap.add_argument("--speed-method", choices=["ffmpeg", "librosa"], default="ffmpeg",
                    help="速度変更の方式。ffmpeg(atempo)推奨。")

    ap.add_argument("--style-prompt", default="")
    ap.add_argument("--style-map")
    ap.add_argument("--style-resolver", choices=["rule", "llm", "hybrid"], default="rule",
                    help="style-prompt を数値化する方式。rule=既存辞書、llm=LLM、hybrid=両者を合成")
    ap.add_argument("--llm-endpoint", default="http://127.0.0.1:1234/v1/chat/completions",
                    help="OpenAI互換 Chat Completions API endpoint（LM Studio等）")
    ap.add_argument("--llm-model", default=None,
                    help="LLMモデル名。未指定時は local-model を使用")
    ap.add_argument("--llm-api-key", default=None,
                    help="LLM APIのBearerトークン（必要な場合のみ）")
    ap.add_argument("--llm-feedback", default="",
                    help="LLM数値化時に渡す主観評価メモ（例: こもる/泥水/苛立ち弱い）")
    ap.add_argument("--llm-timeout", type=float, default=20.0,
                    help="LLM API timeout seconds")
    ap.add_argument("--llm-temperature", type=float, default=0.1,
                    help="LLM temperature")
    ap.add_argument("--llm-max-tokens", type=int, default=200,
                    help="LLM max_tokens")
    ap.add_argument("--llm-debug-file", default=None,
                    help="LLMの入出力デバッグJSON保存先")
    ap.add_argument("--pron-map",
                    help="読み置換辞書(JSON)。例: {\"AI整体師\":\"エーアイ整体師\"}")
    ap.add_argument("--pron-map-to-ref-text", dest="pron_map_to_ref_text", action="store_true", default=True,
                    help="pron-map を参照テキスト(ref_text)にも適用する（デフォルト: ON）")
    ap.add_argument("--no-pron-map-to-ref-text", dest="pron_map_to_ref_text", action="store_false",
                    help="pron-map の参照テキスト(ref_text)への適用を無効化する")
    ap.add_argument("--log-style")
    ap.add_argument("--silence", type=float, default=0.25)

    ap.add_argument("--language", default="Japanese")
    ap.add_argument("--out", required=True)

    ap.add_argument("--vc-backend", choices=["none", "external_cmd"], default="none",
                    help="VC段を使うか。external_cmd は外部VCコマンドを呼び出す")
    ap.add_argument("--vc-input", choices=["generated", "source"], default="generated",
                    help="VC入力。generated=本スクリプト生成音声, source=source-audio を直接VC")
    ap.add_argument("--vc-cmd-template", default=None,
                    help="external_cmd 用コマンドテンプレート。{in_wav}/{out_wav}/{ref_audio}/{source_audio}/{vc_model}/{vc_index} と *_q が使用可")
    ap.add_argument("--vc-model", default=None,
                    help="VCモデルパス（テンプレート置換用）")
    ap.add_argument("--vc-index", default=None,
                    help="VC indexパス（テンプレート置換用）")
    ap.add_argument("--vc-workdir", default=None,
                    help="VC外部コマンド実行ディレクトリ")
    ap.add_argument("--vc-hf-repo", default=None,
                    help="VCモデルをHugging Face Hubから自動DLする repo_id")
    ap.add_argument("--vc-hf-revision", default=None,
                    help="HF repo revision / branch / tag")
    ap.add_argument("--vc-hf-subdir", default=None,
                    help="HF repo内のVCファイル配置サブディレクトリ")
    ap.add_argument("--vc-hf-token", default=None,
                    help="private repo用 HF token")
    ap.add_argument("--vc-hf-local-dir", default="vc_models",
                    help="HFダウンロード先ルート")
    ap.add_argument("--vc-hf-allow-pattern", action="append", default=[],
                    help="snapshot_download の allow_patterns（複数指定可）")
    ap.add_argument("--vc-hf-force-download", action="store_true",
                    help="HFファイルを強制再ダウンロード")
    ap.add_argument("--vc-setup-only", action="store_true",
                    help="VCモデルの自動DL/構成だけ実行して終了")

    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "float16", "bfloat16"])
    ap.add_argument("--attn", default="flash_attention_2")

    ap.add_argument("--debug", action="store_true",
                    help="デバッグ出力（各行の長さ/振幅など）")
    ap.add_argument("--dump-lines", default=None,
                    help="各行の中間wavを保存するフォルダ（例: dump_lines）")
    ap.add_argument("--wav-subtype", default="PCM_16",
                    help="出力wavのsubtype（既定: PCM_16）")

    args = ap.parse_args()

    vc_hf_manifest = None
    if args.vc_hf_repo:
        # デフォルトは repo 全体をDLして、推論スクリプト含む構成を再現する
        allow_patterns = args.vc_hf_allow_pattern if args.vc_hf_allow_pattern else None
        vc_hf_manifest = setup_vc_assets_from_hf(
            repo_id=args.vc_hf_repo,
            local_root=Path(args.vc_hf_local_dir),
            revision=args.vc_hf_revision,
            subdir=args.vc_hf_subdir,
            token=args.vc_hf_token,
            allow_patterns=allow_patterns,
            explicit_model=args.vc_model,
            explicit_index=args.vc_index,
            force_download=bool(args.vc_hf_force_download),
            debug=args.debug,
        )
        if not args.vc_model and vc_hf_manifest.get("model"):
            args.vc_model = vc_hf_manifest["model"]
        if not args.vc_index and vc_hf_manifest.get("index"):
            args.vc_index = vc_hf_manifest["index"]

    if args.vc_setup_only:
        if vc_hf_manifest is None:
            raise RuntimeError("--vc-setup-only には --vc-hf-repo が必要です")
        print("vc setup done")
        print("manifest:", vc_hf_manifest.get("manifest"))
        print("model:", vc_hf_manifest.get("model"))
        print("index:", vc_hf_manifest.get("index"))
        return

    if args.vc_input == "source" and args.vc_backend == "none":
        raise RuntimeError("--vc-input source を使うには --vc-backend を指定してください")

    source_audio_path = Path(args.source_audio) if args.source_audio else None
    ref_audio_path = Path(args.ref_audio) if args.ref_audio else None

    # Pure VC mode: source audio -> VC -> out
    if args.vc_backend != "none" and args.vc_input == "source":
        if source_audio_path is None:
            raise RuntimeError("--vc-input source では --source-audio が必須です")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_wav_for_vc = ensure_wav_for_option(
                source_audio_path, td, tag="vc_source", sr=None, debug=args.debug
            )
            ref_wav_for_vc = (
                ensure_wav_for_option(ref_audio_path, td, tag="vc_ref", sr=None, debug=args.debug)
                if ref_audio_path else None
            )
            run_vc_backend(
                backend=args.vc_backend,
                in_wav=src_wav_for_vc,
                out_wav=out_path,
                source_audio=src_wav_for_vc,
                ref_audio=ref_wav_for_vc,
                vc_cmd_template=args.vc_cmd_template,
                vc_model=args.vc_model,
                vc_index=args.vc_index,
                vc_workdir=args.vc_workdir,
                debug=args.debug,
            )
        print("saved:", args.out)
        return

    if ref_audio_path is None:
        raise RuntimeError("--ref-audio が必要です（VC source直変換モード以外）")

    # style map
    if args.style_map:
        style_map = load_json_object(Path(args.style_map), "style-map")
    else:
        style_map = DEFAULT_STYLE_MAP

    # pronunciation map
    pron_rules: List[Tuple[str, str]] = []
    if args.pron_map:
        pron_rules = compile_pron_map(load_json_object(Path(args.pron_map), "pron-map"))
        if not pron_rules:
            print(f"[warn] pron-map is empty: {args.pron_map}", file=sys.stderr)

    # resolve style
    llm_style_params = None
    llm_error = None
    if not (args.style_prompt or "").strip():
        style_params = {"pitch": 1.0, "energy": 1.0, "speed": 1.0}
        style_mode = "default"
    elif is_numeric_dsl(args.style_prompt):
        if args.style_resolver in ("llm", "hybrid"):
            print("[warn] style-prompt がDSL形式のため、LLM resolver は使わずDSLを優先します。", file=sys.stderr)
        style_params = clamp_style_params(parse_numeric_dsl(args.style_prompt))
        style_mode = "dsl"
    else:
        rule_params = clamp_style_params(nl_prompt_to_params(args.style_prompt, style_map))
        style_params = rule_params
        style_mode = "nl-rule"
        if args.style_resolver in ("llm", "hybrid"):
            llm_model = args.llm_model or "local-model"
            try:
                llm_style_params = llm_prompt_to_params(
                    prompt=args.style_prompt,
                    style_map=style_map,
                    endpoint=args.llm_endpoint,
                    model=llm_model,
                    api_key=args.llm_api_key,
                    timeout_sec=float(args.llm_timeout),
                    temperature=float(args.llm_temperature),
                    max_tokens=int(args.llm_max_tokens),
                    feedback=args.llm_feedback,
                    debug_file=args.llm_debug_file,
                )
                if args.style_resolver == "llm":
                    style_params = llm_style_params
                    style_mode = "nl-llm"
                else:
                    style_params = merge_style_params(rule_params, llm_style_params)
                    style_mode = "nl-hybrid"
            except Exception as e:
                llm_error = str(e)
                raise RuntimeError(
                    "LLM style resolve failed. "
                    "style-resolver=llm/hybrid のため処理を中断します。\n"
                    f"endpoint={args.llm_endpoint}\n"
                    f"model={llm_model}\n"
                    f"error={llm_error}"
                ) from e
    style_params = clamp_style_params(style_params)
    if args.debug:
        print(f"style-mode: {style_mode}  resolved={json.dumps(style_params, ensure_ascii=False)}")

    # read ref_text
    ref_text = None
    if args.ref_text_file:
        ref_text = read_text_flexible(Path(args.ref_text_file), label="ref-text-file").strip()

    # read lines
    source_text_auto = None
    if args.text_file:
        raw_lines = read_lines(Path(args.text_file))
        if args.source_audio and args.debug:
            print("[info] --text-file が指定されているため、source-audio文字起こしは本文に使いません。")
    elif args.source_text_file:
        raw_lines = read_lines(Path(args.source_text_file))
    elif args.source_audio:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_wav = td / "source.wav"
            ffmpeg_to_wav(Path(args.source_audio), src_wav, sr=16000, debug=args.debug)
            source_text_auto = whisper_transcribe(
                src_wav,
                language=args.source_whisper_language,
                model_size=args.whisper_model,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                beam_size=args.whisper_beam_size,
                debug=args.debug,
            )
        source_text_auto = (source_text_auto or "").strip()
        raw_lines = [source_text_auto] if source_text_auto else []
        if args.source_text_out:
            Path(args.source_text_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.source_text_out).write_text(source_text_auto, encoding="utf-8")
    else:
        raise RuntimeError(
            "入力テキストがありません。--text-file か --source-text-file か "
            "--source-audio(Whisper起こし) を指定してください。"
        )

    if not raw_lines:
        raise RuntimeError("入力テキストが空です。--text-file / --source-text-file / --source-audio を確認してください。")
    lines = [apply_pron_map(line, pron_rules) for line in raw_lines] if pron_rules else raw_lines
    if args.debug and pron_rules:
        changed = sum(1 for a, b in zip(raw_lines, lines) if a != b)
        print(f"pron-map: entries={len(pron_rules)} changed_lines={changed}/{len(lines)}")

    # prosody refs
    effective_prosody_refs = list(args.prosody_ref or [])
    if args.source_audio and args.source_as_prosody_ref and not effective_prosody_refs:
        effective_prosody_refs = [args.source_audio]
        if args.debug:
            print("[info] source-audio を prosody-ref として自動利用します。")

    prosody_by_phoneme = bool(args.prosody_by_phoneme)
    auto_enabled = False
    if (
        args.source_audio
        and effective_prosody_refs
        and not prosody_by_phoneme
        and not args.no_auto_prosody_by_phoneme
    ):
        prosody_by_phoneme = True
        auto_enabled = True
        if args.debug:
            print("[info] source-audio モードのため prosody-by-phoneme を自動ONにしました。")

    prosody_paths, prosody_weights = (
        parse_weighted_paths(effective_prosody_refs) if effective_prosody_refs else ([], [])
    )
    warn_prosody_settings(
        ref_audio=ref_audio_path,
        prosody_paths=prosody_paths,
        strength=float(args.prosody_strength),
        smooth=int(args.prosody_smooth),
    )

    # log
    if args.log_style:
        entry = {
            "time": datetime.now().isoformat(),
            "style_prompt": args.style_prompt,
            "mode": style_mode,
            "resolved": style_params,
            "style_resolver": args.style_resolver,
            "llm_endpoint": args.llm_endpoint if args.style_resolver in ("llm", "hybrid") else None,
            "llm_model": (args.llm_model or "local-model") if args.style_resolver in ("llm", "hybrid") else None,
            "llm_feedback": args.llm_feedback if args.style_resolver in ("llm", "hybrid") else "",
            "llm_params": llm_style_params,
            "llm_error": llm_error,
            "source_audio": args.source_audio,
            "source_text_file": args.source_text_file,
            "source_text_out": args.source_text_out,
            "source_whisper_language": args.source_whisper_language,
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "whisper_compute_type": args.whisper_compute_type,
            "whisper_beam_size": args.whisper_beam_size,
            "source_as_prosody_ref": bool(args.source_as_prosody_ref),
            "prosody_refs": effective_prosody_refs,
            "prosody_strength": args.prosody_strength,
            "prosody_by_phoneme": prosody_by_phoneme,
            "auto_prosody_by_phoneme": auto_enabled,
            "prosody_segments": args.prosody_segments,
            "prosody_smooth": args.prosody_smooth,
            "pron_map": args.pron_map,
            "pron_map_to_ref_text": bool(args.pron_map_to_ref_text),
            "vc_backend": args.vc_backend,
            "vc_input": args.vc_input,
            "vc_model": args.vc_model,
            "vc_index": args.vc_index,
            "vc_workdir": args.vc_workdir,
            "vc_hf_repo": args.vc_hf_repo,
            "vc_hf_revision": args.vc_hf_revision,
            "vc_hf_subdir": args.vc_hf_subdir,
            "vc_hf_local_dir": args.vc_hf_local_dir,
            "vc_hf_manifest": vc_hf_manifest.get("manifest") if vc_hf_manifest else None,
            "model": args.model,
        }
        Path(args.log_style).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_style, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # prepare prosody targets (segment profiles) based on FIRST line's segment count unless overridden
    tgt_f0_means = None
    tgt_rms_means = None
    first_text = lines[0]
    N = int(args.prosody_segments) if int(args.prosody_segments) > 0 else len(g2p(first_text) or fallback_units(first_text))
    N = max(1, N)

    if prosody_paths:
        if args.debug:
            print(f"[stage] analyzing prosody refs: {len(prosody_paths)} file(s)")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            f0_profiles = []
            rms_profiles = []
            need_f0_profile = not bool(args.no_world_pitch)

            for i, p in enumerate(prosody_paths):
                w = td / f"prosody_{i}.wav"
                ffmpeg_to_wav(Path(p), w, sr=16000)
                y, sr = librosa.load(str(w), sr=16000, mono=True)
                dur = float(len(y) / sr)
                segs = make_uniform_segments(N, dur)
                if need_f0_profile:
                    f0, t = extract_f0(y.astype(np.float32), sr)
                    f0_profiles.append(segment_f0_means(f0, t, segs))
                rms_profiles.append(segment_rms_means(y.astype(np.float32), sr, segs))

            tgt_f0_means = blend_profiles(f0_profiles, prosody_weights) if need_f0_profile else None
            tgt_rms_means = blend_profiles(rms_profiles, prosody_weights)

    # convert ref audio to wav and optionally whisper
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref_wav = td / "ref.wav"
        ffmpeg_to_wav(ref_audio_path, ref_wav, sr=16000)

        if args.whisper_ref_text:
            if args.debug:
                print("[stage] transcribing ref-audio with Whisper...")
            ref_text = whisper_transcribe(
                ref_wav,
                language="ja",
                model_size=args.whisper_model,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                beam_size=args.whisper_beam_size,
                debug=args.debug,
            )
            if args.ref_text_out:
                Path(args.ref_text_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.ref_text_out).write_text(ref_text, encoding="utf-8")
        if ref_text and pron_rules and args.pron_map_to_ref_text:
            ref_text = apply_pron_map(ref_text, pron_rules)

        if not ref_text:
            raise RuntimeError("ref_text がありません。--ref-text-file か --whisper-ref-text を指定してください。")

        # load model
        if args.debug:
            print("[stage] loading qwen3-tts model...")
        model = load_qwen3_tts(args.model, args.device, args.torch_dtype, args.attn)
        if args.debug:
            print("[stage] generating lines...")

        wavs = []
        out_sr = None

        for line in lines:
            y, sr = generate_voice_clone(
                model=model,
                text=line,
                language=args.language,
                ref_audio_wav=str(ref_wav),
                ref_text=ref_text,
            )
            out_sr = sr

            # speed first (optional)
            _speed = float(style_params["speed"]) if args.speed_override is None else float(args.speed_override)
            if args.no_speed:
                _speed = 1.0
            y = apply_speed(y, sr, _speed, method=args.speed_method)

            if prosody_by_phoneme:
                y = apply_pitch_energy_segmentwise(
                    y=y,
                    sr=sr,
                    text=line,
                    tgt_f0_means=tgt_f0_means,
                    tgt_rms_means=tgt_rms_means,
                    strength=float(args.prosody_strength),
                    style_pitch=float(style_params["pitch"]),
                    style_energy=float(style_params["energy"]),
                    smooth_segments=int(args.prosody_smooth),
                    override_segments=int(args.prosody_segments),
                do_world_pitch=(not args.no_world_pitch),
                )
            else:
                # global fallback
                if args.no_world_pitch:
                    # WORLDの分析/合成を通すと音色がレトロ/カセット寄りになりやすいので完全スキップ
                    y = np.clip(y * float(style_params["energy"]), -1.0, 1.0).astype(np.float32)
                else:
                    y64 = y.astype(np.float64)
                    f0, t = pw.harvest(y64, sr)
                    f0 = f0.astype(np.float32)
                    f0[f0 < 50] = 0
                    f0_orig = f0.copy()
                    voiced = f0 > 0
                    if voiced.sum() > 5:
                        f0[voiced] *= float(style_params["pitch"])
                        sp = pw.cheaptrick(y64, f0_orig.astype(np.float64), t, sr)
                        ap = pw.d4c(y64, f0_orig.astype(np.float64), t, sr)
                        y = pw.synthesize(f0.astype(np.float64), sp, ap, sr).astype(np.float32)
                    y = np.clip(y * float(style_params["energy"]), -1.0, 1.0).astype(np.float32)
            if args.debug:
                dur = float(len(y) / sr) if sr else 0.0
                peak = float(np.max(np.abs(y))) if len(y) else 0.0
                print(f"line: {dur:.2f}s  samples={len(y)}  peak={peak:.3f}")
            if args.dump_lines:
                outd = Path(args.dump_lines)
                outd.mkdir(parents=True, exist_ok=True)
                idx = len(wavs)
                sf.write(str(outd / f"line_{idx:03d}.wav"), y.astype(np.float32), sr, subtype="PCM_16")
            wavs.append(y.astype(np.float32))

    final = concat_with_silence(wavs, out_sr, silence=float(args.silence))
    if out_sr is None:
        raise RuntimeError("sampling rate is None (generation failed?)")

    if final.size < int(out_sr * 0.2):
        raise RuntimeError(f"output too short: samples={final.size} sr={out_sr} sec={final.size/out_sr:.4f}")

    if not np.isfinite(final).all():
        raise RuntimeError("output contains NaN/Inf")

    subtype = (args.wav_subtype or "PCM_16")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.vc_backend == "none":
        sf.write(str(out_path), final, out_sr, subtype=subtype)
        print("saved:", args.out)
    else:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_vc = td / "tts_generated.wav"
            sf.write(str(in_vc), final, out_sr, subtype=subtype)
            source_audio_for_vc = (
                ensure_wav_for_option(source_audio_path, td, tag="vc_source", sr=None, debug=args.debug)
                if source_audio_path else None
            )
            ref_audio_for_vc = (
                ensure_wav_for_option(ref_audio_path, td, tag="vc_ref", sr=None, debug=args.debug)
                if ref_audio_path else None
            )
            run_vc_backend(
                backend=args.vc_backend,
                in_wav=in_vc,
                out_wav=out_path,
                source_audio=source_audio_for_vc,
                ref_audio=ref_audio_for_vc,
                vc_cmd_template=args.vc_cmd_template,
                vc_model=args.vc_model,
                vc_index=args.vc_index,
                vc_workdir=args.vc_workdir,
                debug=args.debug,
            )
        print("saved(vc):", args.out)


if __name__ == "__main__":
    main()
