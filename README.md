# voice_clone_full_pipeline (v3)

音声クローン＋抑揚制御＋外部VC統合のための包括的な音声合成パイプライン。Qwen3-TTSベースの実装を想定しています。

## 主な機能

### v3 で追加された機能
- **音素区間ベースの抑揚（prosody）制御**: 生成テキストを音素単位に分割し、参照音声から抽出した各音素区間の F0（ピッチ）と Energy（音量）を段階的に適用
- **G2P統合**: pyopenjtalk による日本語テキストの音素化（利用可能な場合）
- **境界平滑化**: 音素区間の境界で段差が生じないよう、指定したウィンドウ幅で平滑化

### 既存機能
- **Whisper統合**: 参照音声から自動でテキスト起こし（`ref_text` の自動生成）
- **WORLD後処理**: pyworld を使った pitch/energy のフック処理
- **2段階変換**: 元音声の内容+抑揚を使って、声だけ差し替え（`--source-audio` モード）
- **外部VCバックエンド**: RVC等の外部ボイスチェンジャーを後段に接続可能
- **HF Hub統合**: Hugging Face Hub からVCモデルを自動ダウンロード・セットアップ
- **発音辞書**: 造語・アクセント対策として、読みを固定できる `pron_map` 機能
- **LLMベースのスタイル解決**: 自然言語の感情表現を pitch/energy/speed に変換（`--style-resolver llm`）
- **ハイブリッドモード**: 既存辞書とLLMの推定結果を混合して安定化（`--style-resolver hybrid`）

---

## インストール

### 必須パッケージ
```bash
pip install numpy soundfile torch librosa pyworld
```

### オプション
- **Whisper（音声認識）**:
  ```bash
  pip install faster-whisper
  ```
- **G2P（日本語音素化）**:
  ```bash
  pip install pyopenjtalk
  ```
- **Hugging Face Hub（VCモデル自動DL）**:
  ```bash
  pip install huggingface_hub
  ```

### 外部依存
- **ffmpeg**: 音声ファイル変換に必須。システムの PATH に配置してください。

---

## 基本的な使い方

### 1. シンプルな音声クローン
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --out final.wav
```

### 2. Whisperで参照テキストを自動生成
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --whisper-ref-text \
  --ref-text-out myvoice_ref_auto.txt \
  --text-file input.txt \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --out final.wav
```

---

## 音素区間ベースの抑揚制御（v3）

### 概要
`--prosody-by-phoneme` を指定すると、生成音声を音素単位に分割し、参照音声の各音素区間から抽出した平均ピッチ・平均音量を適用します。これにより、細かな抑揚変化を反映できます。

### 基本例
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --prosody-ref calm.wav:0.7 \
  --prosody-ref irritated.wav:0.3 \
  --prosody-by-phoneme \
  --prosody-strength 1.0 \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --out final.wav
```

**`--prosody-ref` の書式**: `ファイルパス:重み`
- 重みは 0〜1 の範囲で指定
- 複数指定すると、各音素区間ごとに重み付き平均を計算

### 音質調整のヒント

#### こもり・荒れを減らす場合（推奨設定）
```bash
  --prosody-strength 0.2 \
  --prosody-smooth 9 \
  --no-world-pitch
```
- `--prosody-strength`: 抑揚の強度（0〜1）。0.1〜0.3 が推奨
- `--prosody-smooth`: 音素区間境界の平滑化ウィンドウ幅（7〜11 が推奨）
- `--no-world-pitch`: WORLD での pitch 変更を無効化（ピッチ変更による音質劣化を回避）

#### 区間数を固定する場合
```bash
  --prosody-segments 60
```
デフォルトでは、生成テキストの音素数に応じて自動的に区間数が決まりますが、固定値を指定することもできます。

---

## 2段階変換（内容+抑揚 → 声質変換）

`--source-audio` を指定すると、元音声（source）の内容と抑揚を使い、声だけ参照音声（ref）の声質に変換します。

### 自動挙動
1. `--text-file` 未指定時: source 音声を Whisper で文字起こしして本文に使用
2. `--prosody-ref` 未指定時: source 音声を自動的に prosody 参照として利用
3. `--prosody-by-phoneme` も自動的に有効化

### 実行例
```bash
python voice_clone_full_pipeline.py \
  --ref-audio target_voice.wav \
  --ref-text-file target_voice_ref.txt \
  --source-audio source_content_prosody.wav \
  --source-text-out source_auto.txt \
  --style-prompt "ニュートラル" \
  --out converted.wav
```

### 自動挙動を無効化する場合
```bash
  --no-source-as-prosody-ref \
  --no-auto-prosody-by-phoneme
```

---

## 外部VCバックエンド統合

### 概要
`--vc-backend external_cmd` を指定すると、本スクリプト生成後に外部のボイスチェンジャー（RVC等）を呼び出せます。

### 入力モード
- `--vc-input generated`: 本スクリプトで生成した音声にVCを適用（デフォルト）
- `--vc-input source`: `--source-audio` を直接VCに入力（純VC）

### コマンドテンプレート変数
- `{in_wav}`: VC入力wavファイルパス
- `{out_wav}`: VC出力wavファイルパス
- `{source_audio}`: `--source-audio` のパス
- `{ref_audio}`: `--ref-audio` のパス
- `{vc_model}`: `--vc-model` で指定したモデルパス
- `{vc_index}`: `--vc-index` で指定したインデックスパス
- `{*}_q`: 引用符付きバージョン（例: `{in_wav_q}`）

### RVC系スクリプト接続例
```bash
python voice_clone_full_pipeline.py \
  --source-audio source.wav \
  --ref-audio target_ref.wav \
  --vc-backend external_cmd \
  --vc-input source \
  --vc-model rvc_model.pth \
  --vc-index rvc_model.index \
  --vc-cmd-template "python tools/rvc_infer.py --input {in_wav_q} --output {out_wav_q} --model {vc_model_q} --index {vc_index_q}" \
  --out vc_only.wav
```

**注意**: コマンドテンプレートの引数名（`--input`, `--model` など）は、実際に使用する推論スクリプトに合わせて調整してください。

---

## Hugging Face Hub からVCモデルを自動ダウンロード

### 概要
`--vc-hf-repo` を指定すると、HF Hub からVCモデルを自動でダウンロードし、`vc_models/` 配下に展開します。

### セットアップのみ実行
```bash
python voice_clone_full_pipeline.py \
  --vc-hf-repo your-org/your-rvc-model \
  --vc-setup-only
```

### ダウンロード + 純VC を一発実行
```bash
python voice_clone_full_pipeline.py \
  --source-audio source.wav \
  --ref-audio target_ref.wav \
  --vc-hf-repo your-org/your-rvc-model \
  --vc-backend external_cmd \
  --vc-input source \
  --vc-cmd-template "python tools/rvc_infer.py --input {in_wav_q} --output {out_wav_q} --model {vc_model_q} --index {vc_index_q}" \
  --out vc_only.wav
```

### オプション
- `--vc-hf-token`: 非公開リポジトリ用のHFトークン
- `--vc-hf-subdir`: リポジトリ内のサブディレクトリを指定
- `--vc-hf-allow-pattern`: 特定ファイルのみダウンロード（軽量化）
  ```bash
  --vc-hf-allow-pattern "*.pth" \
  --vc-hf-allow-pattern "*.index"
  ```
- `--vc-hf-force-download`: 既存ファイルを強制再ダウンロード

### 自動検出
- `--vc-model` / `--vc-index` 未指定時、リポジトリ内から `.pth` / `.index` ファイルを自動検出

---

## 発音辞書（pron_map）

### 概要
造語や固有名詞のアクセントを固定するため、読みの置換ルールを JSON で定義できます。

### 辞書ファイル例（`pron_map.json`）
```json
{
  "AI整体師": "エーアイ整体師",
  "Qwen3": "キューウェンスリー",
  "TTS": "ティーティーエス"
}
```

### 実行例
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --pron-map pron_map.json \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --out final.wav
```

### デフォルト挙動
- `pron_map` は `--text-file` だけでなく、`ref_text` にも適用されます
- 無効化する場合: `--no-pron-map-to-ref-text`

---

## LLMベースのスタイル解決

### 概要
`--style-resolver llm` を使うと、自然言語の感情表現（`--style-prompt`）を LLM に送信し、`pitch` / `energy` / `speed` の数値を推定させます。

### 対応API
OpenAI互換の Chat Completions API（LM Studio, vLLM, Ollama 等）

### 実行例
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --style-resolver llm \
  --llm-endpoint http://127.0.0.1:1234/v1/chat/completions \
  --llm-model qwen2.5-7b-instruct \
  --llm-feedback "前回はこもる。苛立ちはもう少し強く、速度は落としすぎない" \
  --out final.wav
```

### パラメータ
- `--llm-endpoint`: APIエンドポイント
- `--llm-model`: モデル名（未指定時は `"local-model"`）
- `--llm-api-key`: Bearer トークン（必要な場合）
- `--llm-feedback`: 前回の出力への主観的なフィードバック（例: 「こもる」「泥水っぽい」「苛立ちが弱い」）
- `--llm-timeout`: タイムアウト秒数（デフォルト: 20）
- `--llm-temperature`: 生成温度（デフォルト: 0.1）
- `--llm-max-tokens`: 最大トークン数（デフォルト: 200）
- `--llm-debug-file`: LLMの入出力をJSONで保存（デバッグ用）

### ハイブリッドモード
```bash
  --style-resolver hybrid
```
- 既存の `style_map` 辞書と LLM 推定の中間値を使用
- より安定した結果が得られる場合があります

### エラー処理
- LLM接続に失敗した場合、スクリプトはエラー終了します（フォールバックなし）

---

## スタイルマップの学習

### 概要
`train_style_map.py` を使って、過去の実行ログ（`style_log.jsonl`）から日本語キーワードと数値の対応関係を学習できます。

### 実行例
```bash
python train_style_map.py \
  --logs style_log.jsonl \
  --out style_map.json \
  --min-count 3 \
  --max-keys 200
```

### ログの出力方法
メインスクリプト実行時に `--log-style` を指定：
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --log-style style_log.jsonl \
  --out final.wav
```

### 学習後の使用
```bash
python voice_clone_full_pipeline.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --style-prompt "落ち着いて、ちょっと意味わからない苛立ち" \
  --style-map style_map.json \
  --out final.wav
```

---

## 音素アライメント（拡張ポイント）

### 現在の実装
v3 では、音素区間は **G2P/文字列を全体時間で等分割** する簡易実装です。

### 高精度化
forced alignment 級の精度が必要な場合は、`phoneme_align.py` を以下のツールで置き換えてください：
- Montreal Forced Aligner (MFA)
- Julius
- Kaldi
- その他の音素アライメントツール

### `phoneme_align.py` の仕様
- 入力: 音声ファイル + テキスト
- 出力: JSON形式（音素列 + 各音素の開始・終了時刻）

---

## 高度なオプション

### 速度制御
- `--no-speed`: 速度変更を無効化（time-stretch による音質劣化を回避）
- `--speed-override 1.05`: 速度を強制指定（style-prompt の speed を無視）
- `--speed-method ffmpeg`: 速度変更の方式（`ffmpeg` または `librosa`）
  - `ffmpeg` の atempo フィルタの方が水っぽさが出にくい（推奨）

### ピッチ制御
- `--no-world-pitch`: WORLD での pitch 変更を無効化
  - WORLD 合成を通すと音色がレトロ/カセット寄りになりやすいため、完全スキップしたい場合に使用

### デバッグ
- `--debug`: 詳細ログを出力（各行の長さ、振幅など）
- `--dump-lines dump_lines/`: 各行の中間wavをディレクトリに保存

### 出力形式
- `--wav-subtype PCM_24`: 出力wavのビット深度を変更（デフォルト: `PCM_16`）

### Whisper設定
- `--whisper-model medium`: Whisperモデルサイズ（`tiny`, `base`, `small`, `medium`, `large`）
- `--whisper-device cuda`: 実行デバイス（`auto`, `cpu`, `cuda`）
- `--whisper-compute-type int8`: 量子化タイプ（`auto`, `int8`, `float16`, `float32`）
- `--whisper-beam-size 5`: ビームサーチのビーム幅

### Qwen3-TTS設定
- `--model Qwen/Qwen3-TTS-12Hz-0.6B-Base`: モデルID
- `--device cuda:0`: 実行デバイス
- `--torch-dtype bf16`: データ型（`auto`, `bf16`, `fp16`）
- `--attn flash_attention_2`: Attention実装

---

## 注意事項

### 音素区間の精度
現時点では G2P/文字列を全体時間で等分割した簡易区間です。より高精度な forced alignment が必要な場合は、`phoneme_align.py` を専用ツールで置き換えてください。

### 音質について
- 強い prosody 適用（`--prosody-strength` が大きい）+ 少ない平滑化（`--prosody-smooth` が小さい）は、音質劣化（こもり、荒れ）の原因になります
- 参照音声（`--ref-audio`）と抑揚参照（`--prosody-ref`）に同じ音声を使うと、音色がこもりやすくなります
- 推奨設定: `--prosody-strength 0.1〜0.3` + `--prosody-smooth 7〜11`

### 速度変更の音質
- `--no-speed` で無効化するか、`--speed-override 1.0` で中立にすると、time-stretch による水っぽさを回避できます
- ffmpeg の atempo フィルタは librosa の phase vocoder より音質が良い傾向があります

### ファイル形式
- `--ref-audio`, `--source-audio`, `--prosody-ref` には mp3/mp4 も指定可能です（内部で自動的に wav 化）

### LLM接続
- `--style-resolver llm` または `hybrid` 使用時、LLM接続失敗はエラー終了します（フォールバックしません）
- ローカルLLMサーバー（LM Studio 等）の起動を確認してください

---

## サポートファイル

### `phoneme_align.py`
- 音素アライメントのテンプレート実装
- 現在は G2P + 等分割の簡易版
- forced alignment ツールで置き換え可能

### `train_style_map.py`
- `style_log.jsonl` から style_map を学習
- 日本語キーワード → pitch/energy/speed の対応関係を抽出

### `pron_map.example.json`
- 発音辞書のサンプル
- 造語や固有名詞の読みを固定

---

## ライセンス・クレジット

- 本スクリプトは Qwen3-TTS を想定していますが、他のTTS実装にも対応可能です
- `load_qwen3_tts()` / `generate_voice_clone()` を差し替えてご利用ください
- pyworld, librosa, faster-whisper, pyopenjtalk, huggingface_hub 等の各ライブラリに感謝します

---

## トラブルシューティング

### ffmpeg が見つからない
```
RuntimeError: ffmpeg not found in PATH
```
→ ffmpeg をインストールし、PATH に追加してください

### Whisper が動かない
```
failed to import faster_whisper
```
→ `pip install faster-whisper` を実行してください

### pyopenjtalk が使えない
```
G2P fallback: character-based segmentation
```
→ `pip install pyopenjtalk` を実行してください（任意）
→ インストールしない場合は文字ベースの分割にフォールバックします

### LLM接続エラー
```
LLM request failed
```
→ LLMサーバーが起動しているか確認してください
→ `--llm-endpoint` のURLが正しいか確認してください
→ `--llm-api-key` が必要な場合は指定してください

### 音質がこもる・荒れる
→ `--prosody-strength 0.2` `--prosody-smooth 9` `--no-world-pitch` を試してください
→ 参照音声と抑揚参照に同じ音声を使っていないか確認してください

### 速度変更で水っぽくなる
→ `--no-speed` で無効化するか、`--speed-method ffmpeg` を使用してください
→ `--speed-override 1.0` で速度変更を中立にしてください

---

## お問い合わせ

バグ報告や機能要望は、プロジェクトのissueトラッカーにお願いします。
