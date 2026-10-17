# Ferrum 0.13.0：本轮交付完成

**吞吐优化与默认配置已合并主分支，0.13.0 已正式发布。** GitHub、crates.io 与 Homebrew 已更新；普通 `ferrum run` / `ferrum serve` 按模型能力、设备和容量选择已验证的默认策略，保留显式配置与数值参考路径。

RTX 5090、27B 的八档正式吞吐验收已完成：GGUF 四档超过 llama.cpp；CT INT4 四档按两轮总输出 / 总时长达到 vLLM 的 80%。C32 第二轮为 **78.91%**，不代表每轮稳定达标。完整数据、质量与延迟限制见[结果与决定](performance-popular-models-p0.zh.md)；吞吐达标不等于 SLO 达标。

公开安装已按声明范围验证：Cargo、Homebrew 及 macOS/Linux/Windows 安装入口的版本、身份和启动检查通过；三平台安装后的基础 CPU `run` / `serve` 通过。Linux SM89/SM120 的公网下载超时后，使用原安装器的镜像参数安装与正式资产哈希一致的包，再通过默认公开入口的幂等校验；不将其描述为首次公网完整下载成功。GPU 模型回归使用与正式包一致的已冻结二进制；Windows GPU 模型推理未验收。

中英 README、官网、现有 GitHub Release 与 Homebrew 说明已更新。已发布 crate 内的 README 保留该版本快照，按用户要求不为文档重新发版。5090 临时 runner、代理和相关临时配置已清理；过程与原始证据保存在仓库外。

**下一版本：在事先声明的 TTFT、TPOT、可见 ITL SLO 下最大化吞吐。** 保留当前失败、循环回答和批布局敏感性，不事后放宽阈值；可见 ITL 按相邻非空 SSE 文本事件计算并披露传输合并。后续模型与平台范围另定；Metal 只与 llama.cpp 对照，vLLM 仅在 CUDA 比较。测试与基准继续复用 Rust 工具。
