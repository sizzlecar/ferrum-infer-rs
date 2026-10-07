# Ferrum 目标：先追平CUDA27B，再重评扩展（2026-10-07）

路线已根据[P0实测](performance-popular-models-p0.zh.md)调整：停止扩充原44格，保留13格有效结果，立即对CUDA27B C8的Ferrum/llama差距和Metal C8退化做有界profiling。旧计划、模型选择依据、完整版本与过程证据已随P0页归档；新决定以本文件为准。

## 1. 当前里程碑与长期范围

**首个工程里程碑：RTX5090上Qwen3.8-27B，Ferrum在C4/C8/C16/C32同一GGUF、同一冻结ShareGPT负载下，逐档输出吞吐不低于llama.cpp。** 同时报告TTFT、TPOT、可见ITL的P50/P99、显存/内存、错误、样本数、重复及输出质量，不能以吞吐追平隐藏延迟或质量退化。有SLO约束时按约束验收，尚未声明S时保持unknown。

第4周根据热点修复、四档吞吐/延迟和质量证据重新决定后续范围、优先级及工期。撤销原“第2周两个模型全面超过llama、第6周达到vLLM80%”时间承诺，不再承诺整条路线7–9周完成。vLLM尚无同机实测，不能以未经测量的差距制定比例进度。

长期范围仍是Qwen3.8-27B dense与Qwen3.6-35B-A3B MoE，均属于Qwen3.5架构族并含混合注意力。架构识别不等于所有量化/provider组合可运行：35B支持缺口直接进入P1，不等待其P0矩阵完整。Qwen3.5-9B仅用于合并前C4/C8回归，不低于#402基线。Gemma 4和gpt-oss仍留下一版，不扩充模型名单。

目标机器保持RTX5090 32GB与本机M1 Max 32GB；M4 Mac mini16GB只作9B回归。Metal只对照llama.cpp，vLLM仅考虑CUDA。Ferrum/llama必须用同一GGUF；将来Ferrum/vLLM比较必须用同一检查点，不用GGUF对其他格式的比值替代。vLLM比例、35B/Metal扩展及发布承诺由第4周重评确定。

## 2. 立即执行的P0收口

1. **不再填44格。** 保留13格与此前失败，未完成格不估计、不补零。Metal llama C16保留；旧队列已主动SIGTERM，Ferrum C16未完成、不计结果。CUDA64题质量回放已完成，llama/Ferrum均64题、0请求错误、各45题length截断，guard退出0且原服务恢复核验通过；这不代表质量通过，已见语义错误，原始输出ID为14/64整序列相同、10,366/27,868位置匹配（37.1968%），但64对缺实际prompt ID对齐证据，不能用于证明数值正确或错误。接下来转profile，不启动已备CT短测或探索。
2. **CUDA先做27B C8 Ferrum对llama profile。** 固定硬件、GGUF、容量和样本，分别观察prefill/decode的kernel名称、调用次数、GPU耗时、每轮launch、host等待与同步，形成实测耗时排序。Q4_K路径是待查候选，不是已证实主因；vLLM未测，不猜其差距。
3. **Metal定位Ferrum C8为何比C4慢。** 比较同引擎C4/C8及llama C8，核查batch、dispatch、等待与内存行为。trace有时间/磁盘上限，保存工具可见范围；encoder或整图耗时不能冒充单kernel耗时，deadline差异与失败随比较披露。
4. **35B支持缺口进入P1。** Metal混合MoE不接受目标GGUF的Q6_K专家是实际启动失败；CUDA AWQ/混合BF16专家是格式/provider缺口，未实跑部分仍标未验证。先记录，不在P0零散修补后重填矩阵。

P0退出依据改为：一页现有结果、可复核热点排序、64题输出质量与同GGUF token比较、明确的P1优先级。下载完整、镜像可inspect或短测通过不能替代这些结果；不等35B/vLLM全部就绪才定位已测GGUF差距。

## 3. 对照与复测口径

继续固定ShareGPT、pool208、seed42及选择器；输入4–1024 token，输入加参考输出与模板预留不超过2048；参考输出长度、ignore_eos、thinking off、temperature0。不同并发取同一冻结选择前缀，同格跨引擎样本一致。探索预热8、正式max(32,4×C)、一次重复；验收CUDA每格200正式、两次重复，Metal扩展若恢复则正式max(64,4×C)、两次重复。预热独立计数，profiler采集不混入主负载性能。

CUDA固定32槽，当前27B Metal固定16槽；每请求2048、FP16 KV、全GPU，扫描期间不随并发改容量。llama开启continuous batching/flash attention，关闭prompt cache；Ferrum动态KV与llama共享KV差异如实披露。不改系统内存上限，不把CPU卸载混入全GPU基线。采集前核对实际放置，暂停同机重下载、编译等干扰，由既有guard独占生命周期并恢复原服务。

TTFT取首个可见输出；TPOT截止最后可见文本，按usage输出token减一归一；主ITL为相邻非空SSE文本事件间隔。保留可见停顿、披露coalescing，严格单token资格另作诊断。吞吐只计成功usage输出tokens并除以完整正式窗口；错误、拒绝、样本和内存采样范围同时报告。超时边界不是SLO阈值。

质量回放保持每模型固定64题、贪心和自然EOS；保存输出、停止原因、截断与逐题审阅。HTTP成功、可读文本或完整ID不证明语义正确。两边完整真实ID及匹配prompt齐备后才报告token一致率，不通过重新分词补造ID；截断不计完整答案通过。不同格式不要求逐token相同，但须披露量化差异并检查语义质量。

## 4. P1与后续SLO-aware

P1围绕profile证实的热点成批修复，覆盖受影响的run与serve；支持缺口单独标记。CUDA27B四档同GGUF追平是首个检查点。CUDA Graph、量化矩阵、GDN合批、host同步等只是待测方向，只有热点证据与同硬件复测才能决定优先级。Metal C8退化和35B支持进入同一清单，不能用额外支持项替代首里程碑进展。

第4周评审完整四档吞吐比、尾延迟、错误、质量及回归，再决定35B扩展、vLLM比较和自动SLO配置。达不到预期则重评目标与路线，不追加无profile依据的零散补丁，不在尚无vLLM实测时预设第6周比例。

长期保留完整SLO-aware方向，沿用[既有SLO目标](goal-slo-throughput.zh.md)的约束形式：

```text
G(θ) = max over C of 成功输出tokens / 测量时间
约束：P99(TTFT) ≤ S_TTFT，P99(TPOT) ≤ S_TPOT，P99(可见SSE ITL) ≤ S_ITL，错误 = 拒绝 = 0
目标：max over θ of G(θ)
```

进入该阶段前按已定规则冻结每模型/机器的S：S_TPOT取llama单请求TPOT的2倍，S_ITL取S_TPOT的3倍，S_TTFT取最长prompt单请求prefill时间加事先声明的余量。不得看到Ferrum结果后改S。当前无新模型S，不宣称SLO通过。自动选配置、容量管理、调度及相对手调/vLLM的验收比例和时间，留第4周后按证据重新确定。

## 5. 工作规则

先测后改，同一文件/样本、同机复测，不降低质量或换容易样本。共享推理改动覆盖run与serve，入口特定改动说明范围；按架构/声明能力处理，禁止模型名特例。可复用测试与基准逻辑写Rust，复用已有缓存，不搭建新的包装或认证框架。合并前完成相应正确性与后端检查，编译、协议、语义和性能结论分开。

仓库仅保留本目标页与P0一页结果/决定；完整命令、pins、日志、profile、失败历史和旧文档快照在P0页的库外证据根。目标未达成前不把局部成功写成完成，不承诺未经测量的工期或比例。
