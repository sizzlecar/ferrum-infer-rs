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

# L5 benches: Coder-30B retry (dense-fallback router fix) + the three
# 32B models gates4 skipped.
l5run() { # alias rid repodir csweep
  local alias="$1" rid="$2" repodir="$3" csweep="$4"
  pkill -f "[f]errum serve" 2>/dev/null; sleep 2
  local tokjson tokdir
  tokjson=$(ls /root/.cache/huggingface/hub/"$repodir"/snapshots/*/tokenizer.json 2>/dev/null | head -1)
  if [ -z "$tokjson" ]; then echo "=== STAGE l5_${rid} FAIL (no tokenizer)"; return; fi
  tokdir=$(dirname "$tokjson")
  nohup "$BIN" serve "$alias" --port 8231 > "$W/serve_l5_${rid}.log" 2>&1 &
  local srv=$!
  local healthy=0
  for i in $(seq 1 150); do
    if curl -sf --max-time 2 http://127.0.0.1:8231/health >/dev/null 2>&1; then healthy=1; break; fi
    if ! kill -0 "$srv" 2>/dev/null; then break; fi
    sleep 3
  done
  if [ "$healthy" != "1" ]; then echo "=== STAGE l5_${rid} FAIL (server)"; kill "$srv" 2>/dev/null; return; fi
  "$BIN" bench-serve --base-url http://127.0.0.1:8231 --model "$alias" \
    --tokenizer "$tokdir" --concurrency-sweep "$csweep" --num-prompts 100 \
    --n-repeats 3 --out "$G/l5_${rid}_cuda.json" > "$G/l5_${rid}_run.log" 2>&1 \
    && echo "=== STAGE l5_${rid} PASS" || echo "=== STAGE l5_${rid} FAIL"
  kill "$srv" 2>/dev/null
}
l5run qwen3-coder:30b-gptq coder-30b-gptq models--jart25--Qwen3-Coder-30B-A3B-Instruct-Int4-gptq 1,4,16,32
l5run deepseek-r1:32b-gptq r1-32b-gptq models--OPEA--DeepSeek-R1-Distill-Qwen-32B-int4-gptq-sym-inc 1,4,16,32
l5run qwen3:32b-gptq qwen3-32b-gptq models--JunHowie--Qwen3-32B-GPTQ-Int4 1,4,16,32
l5run qwen2.5-coder:32b-gptq qwen25-coder-32b-gptq models--Qwen--Qwen2.5-Coder-32B-Instruct-GPTQ-Int4 1,4,16,32

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
