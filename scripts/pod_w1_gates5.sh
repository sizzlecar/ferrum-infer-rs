#!/usr/bin/env bash
# W1 single-pod follow-up pass: re-smokes with autosizer-managed pools
# (SMOKE_NO_KV_PIN), the Coder-30B L5 retry after the Marlin-tile dense
# fallback, the R1-8B BF16 smoke, and the L1 fork-point logit-margin
# quantification. Run from the repo root after gates4 + a rebuild at
# >= 16c99602.
set -ux
cd "$(dirname "$0")/.."
. "$HOME/.cargo/env" 2>/dev/null || true
BIN=target/release/ferrum
W=/workspace/w1
G=$W/gates
mkdir -p "$G"
export HF_HUB_OFFLINE=1

smoke5() { # alias rid extra...
  local alias="$1" rid="$2"; shift 2
  pkill -f "[f]errum serve" 2>/dev/null; sleep 2
  FERRUM_BIN=$BIN SMOKE_REQ_TIMEOUT=900 SMOKE_NO_KV_PIN=1 bash scripts/model_coverage_smoke.sh \
    "$alias" "$@" --port 8230 > "$G/smoke5_${rid}_cuda.log" 2>&1 \
    && echo "=== STAGE smoke5_${rid} PASS" || echo "=== STAGE smoke5_${rid} FAIL"
}
smoke5 qwen3-coder:30b-gptq coder-30b-gptq
smoke5 deepseek-r1:32b-gptq r1-32b-gptq --reasoning
smoke5 qwen3:32b-gptq qwen3-32b-gptq
smoke5 deepseek-r1:8b r1-8b-bf16 --reasoning

# Coder-30B L5 retry (dense-fallback router fix)
pkill -f "[f]errum serve" 2>/dev/null; sleep 2
TOKDIR=$(dirname "$(ls /root/.cache/huggingface/hub/models--jart25--Qwen3-Coder-30B-A3B-Instruct-Int4-gptq/snapshots/*/tokenizer.json | head -1)")
nohup "$BIN" serve qwen3-coder:30b-gptq --port 8231 > "$W/serve_l5_coder2.log" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  curl -sf --max-time 2 http://127.0.0.1:8231/health >/dev/null 2>&1 && break
  sleep 3
done
"$BIN" bench-serve --base-url http://127.0.0.1:8231 --model qwen3-coder:30b-gptq \
  --tokenizer "$TOKDIR" --concurrency-sweep 1,4,16,32 --num-prompts 100 \
  --n-repeats 3 --out "$G/l5_coder-30b-gptq_cuda.json" > "$G/l5_coder-30b-gptq_v2.log" 2>&1 \
  && echo "=== STAGE l5_coder_v2 PASS" || echo "=== STAGE l5_coder_v2 FAIL"
kill "$SRV" 2>/dev/null

# L1 fork-point margin quantification: at each divergence point, how big
# is HF's own top1-top2 logit gap? BF16-noise-sized gaps justify the
# self-fingerprint amendment; large gaps mean a real ferrum numerical bug.
pkill -f "[f]errum serve" 2>/dev/null; sleep 2
python3 - > "$G/l1_fork_margins.json" 2> "$G/l1_fork_margins.log" <<'PY'
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

refs = json.load(open("/workspace/w1/gates/l1_refs.json"))
report = json.load(open("/workspace/w1/gates/l1_report.json"))
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda"
)
out = []
for ref, res in zip(refs, report["results"]):
    hf_text = ref["text"]
    fer_text = (res["ferrum_reasoning"] or "") + ("</think>" if res["ferrum_content"] else "") + (res["ferrum_content"] or "")
    # common char prefix -> approximate fork; re-tokenize prefix for ids
    i = 0
    for a, b in zip(hf_text, fer_text):
        if a != b:
            break
        i += 1
    prefix = hf_text[:i]
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": ref["prompt"]}],
        add_generation_prompt=True, return_tensors="pt",
    )
    if not torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids["input_ids"]
    prefix_ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([prompt_ids, prefix_ids], dim=1).to("cuda")
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top2 = torch.topk(logits.float(), 2)
    gap = (top2.values[0] - top2.values[1]).item()
    out.append({
        "prompt": ref["prompt"],
        "fork_char": i,
        "top1": tok.decode([top2.indices[0].item()]),
        "top2": tok.decode([top2.indices[1].item()]),
        "logit_gap": gap,
    })
    print(f"[margin] {ref['prompt'][:32]!r} fork@{i} gap={gap:.4f}", flush=True)
json.dump(out, open("/workspace/w1/gates/l1_fork_margins.json", "w"), indent=1)
gaps = sorted(x["logit_gap"] for x in out)
print(f"median_gap={gaps[len(gaps)//2]:.4f} max={gaps[-1]:.4f} min={gaps[0]:.4f}")
PY
echo "=== STAGE l1_margins DONE"
echo "=== GATES5 COMPLETE"
