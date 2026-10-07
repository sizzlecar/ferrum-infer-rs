# Ferrum 目标：先追平CUDA27B，再重评扩展（2026-10-07）

依据[P0结果与决定](performance-popular-models-p0.zh.md)转入P1修复。热点、数值与历史留结果页及库外证据；目标尚未完成。

## 1. 首里程碑与范围

**RTX5090上Qwen3.8-27B：Ferrum在C4/C8/C16/C32使用同一GGUF、同一冻结ShareGPT负载，逐档输出吞吐不低于llama.cpp。** 同时报告TTFT、TPOT、可见ITL的P50/P99、显存/内存、错误、样本、重复及质量，不能用吞吐追平掩盖延迟或质量退化；有SLO时按约束验收，未声明S时保持unknown。

第4周按四档吞吐、尾延迟、质量与回归重评范围、优先级及工期。撤销“第2周两个模型全面超过llama、第6周达到vLLM80%”承诺；整条路线不再承诺7–9周，vLLM未同机实测不设比例进度。

长期范围仍为Qwen3.8-27B dense与Qwen3.6-35B-A3B MoE，均属Qwen3.5架构族。35B量化/provider支持缺口直接进入P1，不等待P0矩阵完整；Gemma 4、gpt-oss留下一版。Qwen3.5-9B只作合并前C4/C8回归，不低于#402基线。

机器保持RTX5090 32GB与M1 Max 32GB，M4 Mac mini16GB只作9B回归。Metal只对照llama.cpp，vLLM仅考虑CUDA。Ferrum/llama须用同一GGUF；将来Ferrum/vLLM须用同一检查点，不以跨格式比值替代。35B/Metal扩展、vLLM与发布安排留第4周重评。

## 2. 当前P1方向

CUDA R5 C8为65.84 tok/s，仍仅llama的29.23%；较R2吞吐+16.28%，TPOT和可见ITL P99下降，但TTFT P99上升。下一步重做候选profile，再验证量化线性核与大M prefill；单次探索不能称显著或全面改善。

Metal R2 C4近中性，C8吞吐改善但仍比C4慢19.84%，TTFT P99略升；实测剩余热点IQ4_XS的M8路径优先，prefill阻塞另查。encoder计时不是逐shader时间，长负载内存压力仍未排除。

35B未实跑的组合仍标未验证，支持工作不替代27B首里程碑。不恢复44格或启动CT探索，不以准备工作代替同负载收益与质量证据。

## 3. 对照、复测与质量口径

固定ShareGPT、pool208、seed42及选择器；输入4–1024 token，输入加参考输出和模板预留不超过2048。性能负载按参考输出长度、ignore_eos、thinking off、temperature0；不同并发取同一冻结选择前缀，同格跨引擎样本一致。

探索：预热8、正式max(32,4×C)、一次重复。验收：CUDA每格200正式、两次重复；Metal扩展若恢复，正式max(64,4×C)、两次重复。预热独立计数，profiler及合成诊断不混入主ShareGPT结果。

CUDA固定32槽，当前27B Metal固定16槽；每请求2048、FP16 KV、全GPU，扫描不随并发改容量。llama开启continuous batching/flash attention，关闭prompt cache；披露Ferrum动态KV与llama共享KV差异。不改系统内存上限，不把CPU卸载混入全GPU基线；采集前确认放置、暂停同机编译与重下载，由既有guard独占生命周期并恢复原服务。

TTFT取首个可见输出；TPOT为首末可见文本时间差除以usage输出token减一；主ITL为相邻非空SSE文本事件间隔。保留可见停顿并披露coalescing，严格单token资格另报。吞吐为成功usage输出tokens/完整正式窗口；同时报告错误、拒绝、样本和内存采样范围。超时边界不是SLO阈值，单次探索不证明显著收益。

质量回放每模型固定64题、贪心、自然EOS；保存原始输出与ID、停止原因、截断和逐题审阅。HTTP成功、可读或完整ID不证明语义正确，截断不算完整答复通过。报告整序列相等及位置匹配；位置分母为各对较长序列之和。缺实际prompt ID对齐时不推断数值正确性，不重新分词补造ID、不静默去掉EOS；不同格式披露量化差异并检查语义，不要求逐token相同。

## 4. SLO与验收纪律

长期保留[既有SLO目标](goal-slo-throughput.zh.md)：最大化输出吞吐，同时满足事先声明的TTFT、TPOT和可见ITL P99阈值，错误与拒绝为0。违反任一必需SLO的吞吐提升不算成功优化；缺阈值或延迟证据不能证明SLO通过。

进入该阶段前冻结每模型/机器的S：S_TPOT取llama单请求TPOT的2倍，S_ITL取S_TPOT的3倍，S_TTFT取最长prompt单请求prefill时间加事先声明余量；不得看到Ferrum结果后改S。当前S未声明，保持unknown。自动配置、容量与调度，以及相对手调/vLLM的验收比例和工期，由第4周按证据重评。

先测后改，保留数值规则，以真实后端正确性、配对微基准、同负载复测和质量验收；共享改动覆盖run/serve，入口特定改动说明范围。按架构/声明能力实现，不加模型名特例；测试与基准用Rust，复用缓存。合并前完成仓库规定检查，区分编译、协议、语义与性能结论。

仓库仅保留本目标页与P0一页结果/决定；命令、pins、日志、profile、失败历史和完整质量记录留库外。局部成功不等于目标完成。
