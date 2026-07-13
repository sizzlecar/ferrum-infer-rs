# G00 Decision Record: M3 Metal 32 GiB Blocker Rejected

## 状态

`Rejected`。本文件保留为事实记录，不是 active amendment，也不是 G00 contract input。

冻结 legacy SHA `cff4c47765ef3259b8a04890187d99c60da86394` 的
`m3-qwen3-30b-a3b/metal` lane 必须执行完整 canonical correctness matrix；不得用
`legacy-metal-unified-memory-capacity` BLOCKED artifact 代替 PASS。

## 否决证据

2026-07-13 的首轮完整 lane 尝试在初始物理 headroom 仅 `14,548,566,016` bytes 的状态下，
执行 `144/782` case 后触发 swap growth。该结果证明当次机器状态不满足安全门，但不能证明固定
32 GiB 硬件无法执行冻结 lane。

随后在独立、受限的冻结 `ferrum serve` 进程上执行真实非流式
`/v1/chat/completions` 推理：

- artifact：`runtime-vnext-g00-m3-metal-blocked-probe-11a6e9a0-20260713`；
- HTTP status=`200`，推理耗时 `6.065309s`；
- physical headroom：`24,575,819,776 -> 17,856,479,232` bytes；
- swap：`2,197,353,922 -> 2,197,353,922` bytes；
- resource violation=`null`，product return code=`0`；
- bounded process group 已完全回收，峰值 group threads=`17`。

因此 canonical blocker collector 正确输出：

```text
FERRUM RUNTIME VNEXT G00 BLOCKED LANE FAIL: blocked product attempt rejected: command_exit_nonzero
```

精确 failure signature 为 `resource-capacity-violation-missing`。这直接否定了“该 lane 在本机必然
容量阻塞”的假设。

## 后续规则

- G00 对 M3 Metal 的唯一可接受状态恢复为 `pass`。
- 完整 lane 仍须在启动和运行中监测 physical headroom、swap、进程数和线程数；低于 `2 GiB`
  或 active swap growth 仍是本次尝试的 REJECT，不能拼接 partial artifact。
- 若完整 lane 再次因资源门失败，必须先记录当次初始 headroom 和占用来源；不得把机器瞬时状态
  直接升级为硬件能力 blocker。
- G08C、G09、G10 的 M3 Metal correctness、performance 和资源标准不变。
