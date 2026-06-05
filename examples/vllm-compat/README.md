# vLLM compatibility examples

Start Ferrum:

```bash
ferrum pull qwen3:0.6b
ferrum serve qwen3:0.6b \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048
```

Then run:

```bash
python3 examples/vllm-compat/openai_python_chat.py
bash examples/vllm-compat/curl_chat_stream.sh
bash examples/vllm-compat/bench_ferrum_vs_vllm.sh http://127.0.0.1:8000 Qwen/Qwen3-0.6B /path/to/tokenizer-dir
```
