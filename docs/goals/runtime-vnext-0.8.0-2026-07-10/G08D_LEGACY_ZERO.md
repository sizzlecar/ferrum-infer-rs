# G08D: Support Disposition 与 Legacy Zero

## 依赖与目标

- 依赖：G08C
- 下游：G08 aggregator、G09
- 目标：迁移补充 lane，处置全部 support rows，并物理删除 legacy decoder/runtime。

## 必需交付

- Qwen3-Coder agent/tool bug-kill lane迁移。
- DeepSeek-R1 reasoning/template lane迁移。
- Llama 8B-class supplemental release lane迁移。
- 每个 `Architecture` variant、alias、support row、Cargo feature、runtime toggle、executor、
  runner 的 migrated/removed 结论。
- 被撤销支持的旧源码、alias、文档声明和 gate 一并删除；不能只从 release binary 隐藏。

## 验收

- G02 historical production mutation kill `15/15 family` 且 `M/M concrete case`。
- support disposition coverage `100%`，未决/waiver `0`。
- legacy Backend adapter、decoder runner、factory、fallback、feature、hidden-env toggle 数量 `0`。
- arch-named backend API `0`；未批准 model/engine backend cfg 相对 G00 减少 `>=80%` 且新增 `0`。
- release binary 和 source tree 的 legacy inventory 均为 `0`。
- Qwen3-Coder、DeepSeek-R1、Llama supplemental actual-model gates PASS。

```text
FERRUM RUNTIME VNEXT G08D LEGACY ZERO PASS: <out_dir>
FERRUM GATE vnext-g08d PASS: <out_dir>
```
