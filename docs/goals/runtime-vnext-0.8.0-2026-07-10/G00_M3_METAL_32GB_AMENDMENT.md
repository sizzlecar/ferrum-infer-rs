# G00 Amendment: M3 Metal 32 GiB Legacy Resource Blocker

## 决策

冻结 legacy SHA `cff4c47765ef3259b8a04890187d99c60da86394` 的
`m3-qwen3-30b-a3b/metal` lane，在固定 `Apple M1 Max / 24 GPU cores / 32 GiB`
硬件上改为 `blocked`，failure class 固定为：

```text
legacy-metal-unified-memory-capacity
```

这只改变 G00 对冻结 legacy 事实的表达，不修改冻结 binary，不删除 M3 Metal 产品目标，也不构成
correctness/performance PASS。G08C、G09、G10 仍必须让 vNext 候选在同一台 Metal 机器上完成 M3
`run`/`serve`、stream、multi-turn、usage、tool/structured output、完整 correctness 和 performance gate。

## 触发证据

2026-07-13 的 canonical 尝试使用冻结 binary SHA256
`2fd49e84a4f26335446450b1b1eefbbca2d9dccc09575ae287ac82dd70fe01e5` 和固定
`Qwen3-30B-A3B-Q4_K_M.gguf` SHA256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`。

- artifact root：`runtime-vnext-g00-m3-metal-final-9044d00e-20260713`；
- `97` 个独立 `ferrum run` case 完成后 swap 未增长；
- 首次冻结版 `ferrum serve` 启动后，lane 共保存 `144/782` case，其中 `35 pass`、
  `109 known-fail`，product nonzero=`0`，release blocker scan=`0`；
- swap 从 `2,047,019,581` 增至 `2,314,794,435` bytes，增长 `267,774,854` bytes；
- 最低物理 headroom 为 `1,971,896,320` bytes，低于 `2,147,483,648` bytes 硬门槛；
- watchdog 因 `swap_growth` 主动终止，不能把这次 partial run 拼接成 PASS。

关键工件 SHA256：

| 工件 | SHA256 |
|---|---|
| `memory-summary.json` | `6d3c51c3c16fb0ba638a29c5d9e94a12360f7e4defe0033e2ac9073bb262db28` |
| `run-status.json` | `1b065ed5bea4df93d3919b7f8e1aba94afec9e2924fe2bdde8974576c1075fd5` |
| `bounded-receipt.json` | `de5a354d627a1691edc1454b6697b28bb6cdbb3f4c3e6c2d12f7b26ee8cfd36e` |
| `execution-plan.json` | `65622dc7703ca59ce36513129db4db95ec466a58df30fc9481ad22ee9ce72dce` |

## Canonical Blocker Gate

`scripts/release/runtime_vnext_blocked_lane.py` 必须重新执行一次独立、短时、受限的冻结
`ferrum serve` 尝试。正式 blocker artifact 必须同时满足：

1. hardware policy=`metal-reference-m1-max-32gb`，device=`Apple M1 Max`，memory=`32 GiB`；
2. legacy SHA、binary SHA、模型 revision/file SHA 与 `models.lock.json` 完全一致；
3. child env 无继承 `FERRUM_*`，产品与 collector 位于同一 bounded process group；
4. 启动前 physical headroom `>=2 GiB`；
5. 产品仍存活时真实观察到 `swap_growth` 或 physical headroom `<2 GiB`；
6. 保存逐样本 memory JSONL、原始 `vm_stat`/swap/pressure、PID/PGID/start identity、
   effective config、stdout/stderr/failure log 和 bounded receipt；
7. 最终 process group 完全回收，skip/waiver=`0`；
8. blocker 只解锁 G00 的 legacy fact freeze，不产生 M3 Metal performance baseline。

证据采集成功的行是 blocked-evidence PASS，不是产品 PASS：

```text
FERRUM RUNTIME VNEXT G00 BLOCKED LANE PASS: m3-qwen3-30b-a3b/metal: <lane.json>
```

## 下游解除条件

G08C 必须在同一 `32 GiB M1 Max` 上至少满足：

- M3 Metal required correctness case `100%` 完成，unexpected failure=`0`；
- `ferrum run` 与 `ferrum serve` 均 PASS，stream `[DONE]` 恰好一次且 usage 恰好一次；
- 测量 cell physical headroom `>=2 GiB` 且 active swap growth=`0`；
- 性能不低于 G09 中锁定的 same-host external/legacy-eligible 目标；
- exact PASS：`FERRUM RUNTIME VNEXT G08C QWEN3 30B A3B PASS: <out_dir>`。

G10 发布仍要求 Metal/CUDA 三主模型完整 correctness 和 performance；本 amendment 不允许进入
release waiver、跳过 M3 Metal 或改用跨硬件性能结论。
