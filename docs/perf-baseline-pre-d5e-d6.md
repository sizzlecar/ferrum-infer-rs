# Pre-D.5e / D.6 Performance Baseline

Measured **before** any Whisper / Bert / Clip / TTS migration work.
Migration code must match or beat these numbers; otherwise the old
path stays default.

**Environment:**
- M-series Mac (Metal), release binary.
- `target/release/ferrum` built at commit `aa5e783` with
  `cargo build --release --features metal -p ferrum-cli --bin ferrum`.
- Architecture state: Phase D.5a + D.5b landed (Backend extensions for
  Bert / Clip; DenseLinear bias). No D.5c/D.5d/D.5e/D.6 migration yet —
  Whisper / TTS / Bert / Clip still go through candle-based executors.

---

## Whisper ASR (`transcribe whisper-turbo`)

**Input:** `~/ferrum-test-data/meituan_reply_16k.wav` (301 s, 5 min,
Chinese meeting audio).

**Command:**
```bash
time target/release/ferrum transcribe whisper-turbo \
  ~/ferrum-test-data/meituan_reply_16k.wav -l zh
```

**Result:**
- Total wall time: **79.15 s**
- Audio duration: 301 s
- **Real-time factor (RTF): 0.263x** (faster than 1x = faster than real
  time; Python torch CPU baseline was 107 s = RTF 0.356x).

**Quality:** transcript matches expected Chinese output including the
"请不吝点赞 订阅 转发 打赏支持明镜与点点栏目" tail (a known tail-hallucination
pre-existing issue — migration must preserve, not fix).

**Migration gate: ≤ 87 s (10% margin).**

---

## Qwen3-TTS synthesis (`tts qwen3-tts`, no ref)

**Input text:** `你好,欢迎使用语音合成系统,今天天气真不错,适合出门散步` (24 chars).

**Command:**
```bash
time target/release/ferrum tts qwen3-tts \
  "你好,欢迎使用语音合成系统,今天天气真不错,适合出门散步" \
  -o /tmp/baseline_tts_synth.wav
```

**Result:**
- Audio produced: 5.12 s
- Wall time: 14.58 s
- **RTF: 2.85x** (takes 2.85 s to synthesise 1 s of audio).

**Quality check (ASR round-trip):**
```
"你好,欢迎使用语音合成系统,今天天气真不错,适合出门散步。中文字幕志愿者 杨茜茜"
```
(trailing hallucinated text is a known pre-existing issue; migration
must preserve this level of output.)

**Migration gate: RTF ≤ 3.14x (10% margin).**

---

## Qwen3-TTS voice clone (`tts qwen3-tts --ref-audio`)

**Input text:** `你好欢迎使用语音合成系统今天天气真不错适合出门散步`
**Ref audio:** `~/ferrum-test-data/female_ref_5s.wav` (5 s female voice).
**Ref text:** `对你协调的结果不认同你不会说表示继续去沟通我要求你`

**Command:**
```bash
time target/release/ferrum tts qwen3-tts \
  "你好欢迎使用语音合成系统今天天气真不错适合出门散步" \
  --ref-audio ~/ferrum-test-data/female_ref_5s.wav \
  --ref-text "对你协调的结果不认同你不会说表示继续去沟通我要求你" \
  -o /tmp/baseline_voice_clone.wav
```

**Result:**
- Audio produced: 4.96 s
- Wall time: 15.82 s
- **RTF: 3.19x**.

**Quality check (ASR round-trip):**
```
"你好,欢迎使用语音合成系统今天天气真不错,适合出门散步今天天气真不错适合出门散步
 适合出门散步请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"
```

Known behaviour:
- Text content for the first pass correct.
- Repetition tail (`今天天气真不错适合出门散步 今天天气真不错...`) is the
  pre-existing voice-clone decode-sampling divergence noted in
  `project_tts_voice_clone_status.md`. **Not a migration target — it
  pre-exists and must not get worse.**

**Migration gate:**
- RTF ≤ 3.51x (10% margin).
- ASR round-trip must contain the full input text (same repetition
  depth allowed; nothing *more* broken).

---

## Reference wav files

Baseline outputs saved to `/tmp/` during this measurement:
- `/tmp/baseline_tts_synth.wav` — no-ref TTS
- `/tmp/baseline_voice_clone.wav` — voice clone with `female_ref_5s.wav`

These are not checked in. After any migration, regenerate into
`/tmp/migration_*.wav` and compare:
- ASR transcript diff (should be identical or within the known
  hallucination tail).
- Spectrogram eyeballing for gross distortion.
- Optionally speaker-embedding cosine against the reference for voice
  clone (should match pre-migration value — current drift is a known
  limitation, not a migration concern).

---

## How to use this document

Before starting D.5c (Bert), D.5d (Clip), D.5e (Whisper), or D.6 (TTS):

1. Re-run the three commands above on the unchanged baseline build.
   Confirm numbers are within ~2% of what's recorded here (hardware
   variance). If they drift much more, investigate before migrating.

2. Run them again after every migration PR. Record numbers. Don't
   flip the default path until the gate is met.

3. If a migration fails the gate (e.g. Metal LayerNorm shader is
   slower than candle's CPU path for Bert), leave the legacy executor
   as default and file the kernel gap for Phase E.
