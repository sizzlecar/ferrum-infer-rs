# Goal：P-H3-Image 原生量化推理与连续批处理

2026-11-08 Dense 小 M 实测收口：原 controller2449137 于04:55:47 UTC结束，CUDA oracle与micro各1项通过，runtime/restore均0；原Qwen恢复，其余16服务未变。完整原stdout已取回，13组/104对计时和全部数值偏差保留；库外 current-dense-small-m-projection/result.json `580b7ee3…`、独审 `1b7576f0…`。候选/原cuBLAS完成时间配对中位比：packed FFN gate/up M1 1.024、M2 0.995；LM head均约0.989；FFN down M1 0.718、M2 0.930；GDN output M1 0.669、M2 0.739。大矩阵没有足够收益，小矩阵为重复热缓存诊断，不能按比例推算整图收益；本轮不推广生产选择器，没有新增端到端提速结论。

数值诊断的退出0只代表原前向误差界、边界及特殊值检查通过。FFN down M2相对rounded-F64的1e-3偏差原cuBLAS为1052/8192，候选为1/8192；该参考不是完整CPU provider合同。只读核实当前F16 GEMM未设置禁止低精度reduction的math-mode标志，而Qwen35数值声明为F32累加；正在准备仅私有fixture的default/accumulator-reduction诊断，尚未更改生产模式或宣称原合同失败。下一性能判断同时复用原Off计时，避免依据Kernel模式额外event开销挑优化点。Graph旧9.778秒停顿与首次按需capture形态相符，但旧包缺少对应API计时，具体调用未定位；graph cache会随cohort lane退休清空。仍不重开replay或把预热移出测量。本文未暂存/提交。

2026-11-08 04:47 UTC：共享扩写 owner 的 process_mapped(staged) 对照完成，**拒绝推广，保留 bounded_streaming + workers4**。同一 ffcb57d7 程序、两 fresh 进程各冷/暖 duplicate-blue B2，8 个有限原 F32 输出及扩写/EOS/reference latent 全等；各 16 次同步 B2、每请求 8 步，不能替代混进度 HTTP 回归。暖批次 71.180698→73.059096 秒（+2.64%），冷批次 59.904660→85.430938 秒；缓存冷构建 20.821255 秒，已经包含在冷批次时间内。暖静态安装 3.543670→10.284291 秒，省下转换但增加 6.798066 秒 staging copy；其余阶段抵消一部分差额，N1/r1 不足以证明稳定回退。RSS/分页未采到；SMI 峰值均 24816MiB，仅为采样下界。独审 `90c7045e…`、最终结果 `aeaba57a…` 存库外 current-process-mapped-shared-owner-ab；原 Qwen 恢复，其余 16 容器不变，不再追加该缓存路线实验。

DenseF16 小行数投影诊断的 5 文件已冻结 `3cbbb829…`，独立完整源码编译在 04:44:04 UTC 正常结束；尚未通过 GPU 数值或性能验证，未接入生产选择器。M1/M2 共享权重读取、F32 累加、F16 存储，不改激活或量化。先完整 F64 oracle 与边界验证，再同 stream/cuBLAS 的交替微基准；单矩阵热缓存结果不能按权重字节加权外推全模型收益。当前没有新增整图提速或主流追平结论，完整目标仍未完成；本文保持本地、不暂存/提交。

2026-11-08 04:26 UTC：计时接线完成实机验证，完整managed GDN provider N1/N2/N5测试通过；原ffcb57d7 profile的两轮duplicate-blue B2共4个有限F32输出哈希一致、EOS与扩写一致，16次同步B2/0混进度，不能替代原late-join资格。原trace sealed1481/drop0；独审 `6a85b648…`、总result `2762c0ba…`（库外 current-expansion-kernel-timing）。measured owner去重后341轮decode：submitted9.724526s、host encode/submit4.854170s，其中enqueue4.079632s（84.04%）、provider node encode0.606364s；60018次eager command内部2.657540s，其余enqueue父区间1.421885s包含事件/循环等。Kernel插桩有扰动，不做旧Off A/B；managed completion/postprocess/readback零样本属于未采集，真实fence wait4.593682s与上述时段重叠，不能相加或推断可省时间。原959MB profile留远端，仅取完整825KB trace与小摘要；Qwen恢复、其余16容器不变。代理在测试结束后触平台用量限制，现已恢复并重新封存结果，未重投GPU任务。

下一步正在实现独立DenseF16小M投影候选与Rust数值/微基准，暂不接生产selector；FFN已原生packed，不能把新算法称为首次B2支持。另只复评已有Fresh+process_mapped(staged)+workers1策略，相对旧R54的理由是现cohort共用一次安装；有限上传本已存在，无需新增source抽象。旧R54缓存负证据、冷构建和物理RSS/分页代价均保留，候选尚无新收益。ConvRot七文件合同闭包已提交/推送 `db96b8b96`：10项不同测试、runtime/workspace alltargets编译及fmt通过，不包含后端执行或性能声明；docs仍未暂存/提交。完整主流性能、最新回归与L20目标未完成。

2026-11-08 01:15 UTC：GDN四文件候选完成5090编译、实际路由1项及原两kernel逐位对照2项，均通过；50对交替微基准完整保留，5组融合/原版CUDA event配对中位比0.487–0.532。每arm128次，双成员[1,1]的event中位2.375648→1.176848ms；两臂都有尖峰，event包含host-fed gap，仅是epilogue局部诊断，不等于整图加速。原Qwen恢复，其余16容器未变；result `b21f5fef…`、独审 `b517e1ef…`（库外 current-gdn-small-n-epilogue）。下一步合并现有完整managed provider测试与一次真实扩写Kernel诊断，不追加该小算子的十图矩阵。计时接线10文件（production约89行）已冻结 `16f2342e…`，沿现显式profile Kernel选择传到B1/B2独立executor；默认Off不变，MTP+Kernel明确拒绝，编译与实机资格仍在进行。现profile最小warm要求意味着两次duplicate-blue B2/4输出，区别于原green43+blue42，不用于性能比较。

2026-11-08 01:08 UTC：基础代码小提交 `f739dac7a` 已推送，仅 F32 GELU 两种算子契约、几何边界及测试共3文件；独立 b11c1ba5 闭包2项测试、runtime all-target check、fmt通过，未包含docs，也不宣称后端或性能提升。GDN小行数 norm→F16 cast 融合4文件已冻结，CUDA harness编译与1项实际路由测试通过；下一步仅原两kernel逐位对照和交替微基准，不先重复十图矩阵。另已确认当前扩写细分计时未接入独立executor，正在沿现typed Kernel诊断入口补接；Off B2已有341波的host_encode_submit合计3.292s/提交总计8.626s，存在GPU重叠，不能当作可直接节省时间。完整性能、回归和L20目标仍未完成。

2026-11-08 00:52 UTC 最新实机结论：同 cohort main TE 视觉复用 v4 完整源码库存 `0b227fb9…`（3778文件，候选14文件 `254b6935…`）完成正式 CUDA check/build，服务 `97d8757d…` 与 consumer `ce83bf60…` 在5090同二进制 On/Off 各5图，10/10 HTTP200、5对PNG/RGBA及10个实际文件复读哈希一致。controller1993377于00:47:29 UTC自然结束，runtime/restore均0，原Qwen同CID恢复ready/idle，其余16容器保留。On sealed trace实际出现同cohort6的producer6 `reused=false`→`reused=true`，保留39,362,560B；停止过程中尚未刷完文件的一次空grep不是缺失证据。

本轮复用**未通过端到端性能门槛，不推广，默认Off**：C1 On38.1560/Off38.7907s；C2首On58.8205/Off55.5434s、末63.8796/60.6071s，吞吐比0.94877。两臂请求先后相反，必须按sample+seed比图；每格N1/r1不能证明稳定回退或加速/P99/SLO。最终独审5对PNG/RGBA/扩写/EOS摘要/refLatent全等，每图8步、各7真实混进度B2；C2两次视觉计算合计2.69566/3.57353s（On少0.87788s，hit自身28.56ms），共享扩写22.5453/19.8815s（On多2.6638s），首初始化9.0512/6.9669s（On多2.0842s）。15个非重叠整数ns区间完整覆盖服务span差额+3.227639s；HTTP末span差额+3.272458s，不能把嵌套阶段相加。库外 current-main-vision-cohort-reuse-ab 原包 `621b6a3a…`，independent-analysis.json `321c368c…`。物理RSS完整采样峰On30.700GB/Off31.105GB，整卡显存两臂均24,850MiB（采样下界）。两臂idle allocated回16MiB且普通Output0支持共享输出已释放，但不是逐owner遥测；committed仍同16MiB trace加304/608B，不声称账本0。

v4修复真实低内存进展边界：producer原PNG backing在取消/结束后可能只由缓存持有，survivor hardpeak现在计入该额外物理backing（同region不重复），避免把自持内存误作可等待释放的外部压力；保原12GiB总限额。真实Reference三项、CLI/trace consumer各一项通过。冻结源码的fmt/check/test/Clippy/Metal均0，7966 passed、79 ignored；结果为 current-main-vision-cohort-reuse/local-validation/workspace-result-v4.json（8743bdfa…）。原v1编译失败与CLI bin零命中记录保留，不冒充测试通过。两次仅删除已完成的历史重复测试可执行文件，合计释放约3.10GB，源/权重/证据未删。

防止重复试验：R52 F16C、R89/R90 best-fit、R94 main weight arena分块及本次计划缓存4→8均没有建立稳定整链收益；现保留原exact allocation cache与main arena。R91关闭扩写decode replay曾消除长停顿，不能仅因Comfy使用CUDA graph就重新打开。最新缓存4→8原包已封5对全等/真实B2，但family metadata序列两臂均17hit/14miss且C1回退3.37%；没有证据证明扩大缓存有价值。MTP one_draft仍Off（71.5%接受、无稳定收益且一组输出差异），不能把Comfy自适应7候选/8验证行的实现当成本实现已具备的能力。下一实现先沿现GDN的小行数计算路径审查融合，不承诺其足以解决扩写差距；完整目标、主流追平、最新8MP/Qwen回归和L20交付仍未完成。本文未暂存/提交/推送。

2026-11-07 19:13 UTC 当前进展：CUDA tail-two greedy head 的独立四文件闭包已提交/推送 `b11c1ba5e`，与 upstream 相同，index 为空，未提交 docs。独立父版本加补丁的 fmt/check/Clippy/Metal 通过，workspace tests 7133通过、79忽略、0失败，正式 CUDA CLI 三 feature 编译及 kernels lib no-run 通过；这些检查仅覆盖该独立闭包，不代替完整 H3/MTP 的整图等价性。原同字节 CUDA helper 算术证据单列。库外 current-mtp-cuda-head-commit-preparation/commit-receipt.json 为最终提交凭证（938f9d11…）。

主线保持 MTP Off，已完成现有计划缓存每阶段容量 4→8 的同二进制十图对照。原 controller1395414 在19:09:32 UTC 自然完成，runtime/restore 均0；原报告 C1 从36.233016→37.453921s（回退3.37%），C2首56.638433→54.594317s、末61.670727→59.601040s。每格仅N1/r1，候选已违反预声明C1不回退门槛，不推广；完整输出配对、缓存事件与准备耗时归因仍待原包独审，不把波动当成缓存命中收益。证据位于库外 current-plan-cache-capacity-ab，未追加样本或重跑。

同一初始化 cohort 的 main TE 原始 F32 视觉输出复用已实现，显式 `main_vision_reuse` 默认Off；与既有扩写器视觉复用分开。v3十四文件 manifest c900f708… 冻结；真实 Reference 最小2项通过，涵盖原请求 Output=0 scope 关闭后幸存者继续、同图命中/异图不命中、独立输入与最终Host/device退款。CLI/bench及完整检查继续，尚无该候选 CUDA 整图/性能证据。后续同二进制开关对照固定缓存4、MTP Off及扩写器视觉复用Off；目标与性能验收/L20均未完成。

2026-11-07 18:36 UTC 补充定位：保持原5bc二进制，仅对blue42进行fresh On/Off各一张、开启现有typed有效提示词捕获；两张HTTP200、EOS、8步及PNG均精确复现原C1。正文唯一差异为On的`retaining its original texture, shape, and highlights`与Off的`... texture, contours, and highlights`，其余全文相同。已定位可观察的选词差异，尚未定位首次不同token的内部hidden/logits或状态原因，不能直接归因浮点误差。On bench因资源采样`skipped_intervals=1`退出非零，原失败保留；Off为0，两次原Qwen恢复均0，不作新计时比较。证据为库外 current-mtp-c1-prompt-diagnostic，原包a378eb55…、result fbae80ba…。MTP继续默认Off，完整目标仍未完成。

纯设备拷贝准入的CUDA-feature无GPU回归及旧module drop-order检查已在devices=[]的原CUDA主机构建/执行流水线退出0，测试源码单独manifest c24c224d…，未重建原5bc服务。它们验证实际准入及额度owner生命周期，不作为设备拷贝算术或整图等价性的新证明。下一笔小提交候选已收敛为4文件CUDA tail-two head闭包，提取后仍需独立HEAD编译验证；未提交docs。

2026-11-07 18:23 UTC 当前实测结论（优先于下方历史记录）：修复纯设备拷贝准入后，源码0ad82bd5…/二进制5bc2b84b…在5090完成同二进制 MTP On→Off 各5图，10/10 HTTP200、8个Rust benchmark join匹配，两条strict trace均sealed/drop0/incomplete0；每臂33个DiT波，其中7个真实B2且混合步骤，首个为[1,0]。MTP实际接受695/972个草稿（71.502%），C2草稿/验证各198波，其中189波双成员。原Qwen恢复成功，其余非Qwen容器未停；完整原始证据与独审在库外 current-mtp-product-ab-v2，independent-analysis.json SHA5e80e156…。

本轮**未通过输出一致性门槛，不推广**：5对中4对有效扩写/PNG/RGBA全同；蓝壶C1的有效扩写On为915B/426a352c…，Off为918B/ab9370e9…，图像也不同。各编辑reference latent全同，EOS/token数量摘要同，但未采完整token数组/hash，不声称完整token序列相同。该对RGB MAE0.68386/255、RMSE2.79272，65.6028%像素至少一通道不同，alpha全同；这是像素诊断，不是语义质量合格判定。后续仅用原binary各fresh服务一张blue42、现typed有效提示词capture做两图最小诊断，不作新的性能比较。

观测Off→On：C1 38.372→37.438s（输出不等）；C2首58.012→57.114s、末63.020→62.087s、吞吐0.031736→0.032213图/s（+1.501%）。每格N1/r1，不能建立P99/SLO或稳定加速；C1扩写改善主要伴加载8.296→5.324s，C2扩写反从20.528→20.799s，不能把整图小幅波动归为MTP解码收益。未进行新Comfy对照、未追平、L20未更新，MTP默认Off。原预算的Host空闲16MiB trace及304/608B额外commit仍非0。当前本地fmt/check/test/Clippy/Metal全0（7961通过/79忽略）；CUDA真实编译与10图另记，新增纯拷贝准入3项CUDA-feature无GPU回归补丁已合active但尚待运行，不冒用上述测试数。docs未暂存/提交/推送。

2026-11-07 18:01 UTC 当前实机状态（优先于下方历史记录）：冻结源码 542bee0b… 的正式 CUDA CLI check/build 和 Rust image_trace 构建均通过，实际二进制 00f41b5c… 已在5090执行第一张 MTP On 烟测。主模型真实 prefill 完成一波，首 token 为40；首 draft 在提交前因 bounded causal command 合同被拒，HTTP500、0 PNG、0 draft/verify/proposal，后续9图未跑。原 Qwen 同容器恢复成功且 ready/idle，其余16个非Qwen容器保持运行。失败原件与 result.json 位于库外 current-mtp-product-ab，不能将本轮记作整图、命中率或性能资格。

源码定位到 concat_columns_f16 的实际工作是两路设备内拷贝，正确申报 compute=0/transfer>0，旧 bounded Compute 门却只认可 compute>0。已最小修正为两类工作都为0才拒绝，保留原模块 owner、managed transfer、禁止旧 host upload/reusable 的检查，并扩充实际 operation/phase/模块/工作数拒绝信息；尚待新二进制实测确认它是否完整解释首烟失败。本地全 workspace test-v1另发现旧源码文件字串断言失败，因绑定helper已移至子模块；移除该文件位置断言、保留其余packed合同与真实provider回归后，该target18/18通过，完整test-v2/Clippy/Metal继续。最新生产修复不因旧CPU通过而自动合格。仍无新MTP提速数据，无主流追平结论，L20未更新，docs未暂存/提交/推送。

2026-11-07 17:47 UTC 当前进展（优先于下方历史记录）：MTP 已接入 run/serve 共用的原初始化/视觉/下游图像流程，新增显式 `prompt_expansion.mtp = "one_draft"`（默认 `off`）。主/草稿一次 union 安装、独立 KV 游标、分块预填充、真实同波 B2、选择提交、EOS/末槽 N1、初始化期间取消脉冲和实际计算后取消、原错误与部分退休凭证均由原 Core/Host owner 承担。完整冷/热初始化及 product adapter 的 CPU 定向 **15/15 通过**（4 wrapper、7 machine/recovery、3 diagnostics、1 assets），CPU 后端不支持 streaming 静态源，误配该能力的首次 14/15 失败完整保留，最终 Fresh 测试不冒称 CUDA streaming 资格。另 Core workspace-policy 回归 1/1、CLI MTP 解析回归 1/1 通过，fmt 通过；同一冻结源码全 workspace check/test/Clippy/Metal 正在执行，尚不能宣称全部通过。

候选前缀计划误把纯 workspace buckets 当 device replay 的修复已独立提交/推送 `a7245b36f`（3 文件，含真实 selected-One 后续算回归），延续分批 push；docs 未暂存/提交/推送。闭环证据位于库外 current-mtp-engine-cohort/{mtp-all-v2,cli-mtp-v1,workspace-policy-regression-v2,fmt-final-v2}.*，小提交receipt为workspace-policy-commit.json。

5090完整源码v2冻结 manifest为542bee0b3603c100ad4ccb652215cd7b52e75b578f7014b1bf85f0db88126810（3776文件），CUDA正式check/build与实际MTP全模型尚待；当前无新生图计时或加速结论。库外 current-mtp-product-ab/preparation 已准备同二进制 On→Off 两臂各5图：生成/编辑暖机资格各1、编辑C1一张、同图C2两张；cap2、29GiB/12GiB、worker4/tile4、CPU noise、visionreuseOff与replayOff固定。唯一推理配置差为MTP；统计仅诊断规模，不能建立P99或SLO稳定性。网络probe经mini曾超时，正优先核上次成功的5090 direct helper；没有新GPU任务或服务停启可虚报。本地仅清4旧测试exe、透明压缩87个本任务Cargo产物（SHA/mtime保留），释放约3.8GB，未动权重/源/证据。主流性能追平、最终当前二进制Qwen实机回归和L20仍未完成。

2026-11-07 17:12 UTC 当前进展（优先于下方历史记录）：末两行 greedy head 已以 `98c3fe243` 独立提交并推送（17 文件，逐 blob 与冻结补丁一致），接续前两笔 selected-prefix/shared-weights 小提交；docs 未提交。主/草稿模型共享一次 union 安装、真实 MainVerify/OneDraft 程序与 managed causal wave 桥接已完成 4 项真实 CPU 集成测试，覆盖 N1/N2/选中前缀续算、B2 取消隔离、实际 proposal 接受/拒绝/EOS、分段 token/RoPE 及错误输入拒绝，原 Host/Device 最终归零。此资格不等于完整扩写循环或整图提速。

CUDA candidate managed v6 1/1 通过，但它在原预算内先按原编译最大值预置池；此前 N1→N2 弹性分配等待问题未解决，不可宣传为弹性资格。tail-two/concat 三项真实 GPU helper 通过，dense fc 的正式 CUDA CLI check 与 kernels lib no-run 通过；这三项不是完整模型 selected wave。原 Qwen 已恢复且最新 compile-only 窗口没有停服，其余 WSL 服务未改。证据：库外 current-mtp-cuda-candidates/result-v6.json、current-mtp-cuda-tail-head/result.json、current-mtp-cuda-dense-fc/compile-result-v2.json。

完整 engine 主/草稿状态机本轮已接入分块 prefill、独立 KV 游标、真实 canonical B2 wave、草稿验证、EOS/末槽 N1、compute 后下一 poll 重读取消再提交，以及各 sequence 独立退休 receipt；联合 engine all-target check 通过，实际 Host/零视觉/位置表测试 1/1 通过。完整循环 Rust fixture 正在补，不把编译当执行资格；尚无新 MTP 全模型/整图计时或 L20 更新。MTP 仅加速扩写解码：既有 4MP 编辑 36.964s 中该段 6.379s，即使该段理想减半，总耗时约33.775s（约8.6%），并非实测收益，不能独自承诺解决 C2 首张延迟。下一步先闭环 CPU、同机实际命中率/解码耗时，再做整图对照。

2026-11-07 16:00 UTC 当前进展（优先于下方历史记录）：共享不可变 token 权重安装已以 `3b18bd934` 独立提交并推送（8 文件），冻结补丁对应两项真实 CPU token 测试、两项 Core subset 回归及五项既有 component 共享测试通过；不代表完整 MTP 已接通。active 已保留原 Host 预算与 PhaseCustody 接入 managed publish/attach 两个入口，旧共享 2 项、Host/custody 10 项与 execution 编译通过，新增直接 managed 入口测试正在运行。bounded tail-two F16 greedy head 与 TerminalOnly 依赖证明已合 active 和模型隔离树，active head 4 项、Core 10 项通过；其 CUDA provider、草稿模型及接受/拒绝执行循环仍未完成。

5090 原始 CUDA candidate scan 的真实 B2 测试已通过（1 项，2.84 秒），覆盖不同进度与只读原状态。完整 managed provider 测试先后在静态上传支持与 canonical 设备标识的测试配置处失败，均尚未进入算术，原失败保留；已对照正式 H3 工厂修正并复核真实 1GiB 预算，最新 v5 正在编译/重测，不能记为通过。此前测试窗口结束后原 Qwen 恢复成功，其余 16 个非 Qwen 容器身份未变。本轮无新增完整生图计时、ComfyUI 对照或 SLO 结论；目标保持进行中，L20 未更新，docs 未暂存、提交或推送。证据为库外 current-mtp-shared-token-installation/commit-receipt.json、current-mtp-bounded-head/active-integration/、current-mtp-cuda-candidates/gpu-original-summary.json。

2026-11-07 15:24 UTC 最新进展（优先于下方历史记录）：选中前缀 Core 事务已以 `bc1fe898a` 单独提交并推送，30 文件闭包包含 provider 显式前缀效果、候选状态保留、原 completion reaper 上的受控复制、成功后前沿提交及视觉/位置输入验证。新增 9 项真实 Core 测试通过，覆盖不同接受长度、取消/失败/未知完成、错误 token 和视觉来源、稀疏零区域及 output-only 不得影响状态；runtime 全库 918 passed/5 ignored。提交对应的隔离工作区完整 fmt/check/test/Clippy/Metal 编译均退出 0，工作区测试为 7125 passed、0 failed、79 ignored；这是提交源码的范围，不能与 active 完整 H3 快照的 7916 项相混或相加。证据为库外 current-mtp-prefix-settlement/{freeze-v1/manifest,workspace-validation/result}.json。

实际 execution 桥接已在 active 接上原 ManagedHostBytes 视觉/位置所有权，按编译计划绑定一次，并在前缀缓存关闭时保留 candidate 必需的精确 token 证据；5 项定向测试及 execution 全目标编译通过，仍未提交依赖尚未整理的 H3 产品代码。CUDA candidate 实现 v2 已冻结，保留原 F16/F32 数值路径、只读原状态及两份尾状态，正在上传做正式编译；v1 的 managed GPU fixture 误用 FixedTensor 入口已在上卡前发现并改为真实 Token Core，原件保留，没有放宽产品限制。主/草稿共享安装及有界末两行 head 在独立工作树并行推进。完整 MTP 的模型、driver、CUDA 实机及生图尚未闭环，本轮没有新增速度结果；当前确认收益仍为同参考图 C2 约 5.96%，ComfyUI 追平和 L20 仍未完成。docs 未暂存、提交或推送。

2026-11-07 14:47 UTC 最新进展（优先于下方历史记录）：包含已推送 CPU 候选状态算子的完整冻结源码 c90753cb…已完成 Linux workspace check/test/Clippy 和正式 CUDA CLI 编译检查，全部退出 0；顶层 297 项 target summary 合计 7916 passed、0 failed、81 ignored，嵌套子进程结果单列。active 完整 Metal 编译亦退出 0。这些是编译/CPU 回归证据，不是新 GPU 推理或提速结果；原 12 图 A/B 使用的二进制未改，Qwen 原进程保持健康空闲。证据为库外 current-post-mtp-source-qualification/validation-summary.json 及 local-metal.log。

后续 MTP 选中前缀提交的 6 项真实 Core 测试已通过，覆盖接受 1/2 个 token 后续算、双成员隔离、取消、部分拷贝失败及未知 fence 保留资源；最初 fixture 违反 causal family 的 Stored 合同已按真实初始化修正，原失败保留。该实现仍在隔离工作树，正补视觉/位置输入的原始受预算管理内存绑定及实际提交前校验，尚未冻结、提交或接入实际生图；不能计作单张或首张提速。当前实测改善仍为下述同参考图双请求约 5.96%，没有新 ComfyUI 对照，性能追平和 L20 仍未完成。docs 不暂存、不提交、不推送。

2026-11-07 14:12 UTC 当前里程碑（优先于下方历史记录）：同 cohort 原图视觉编码复用已完成 5090 上同一不可变二进制 Off/On 12 图对照，6 对实际 PNG/RGBA、扩写文本及 EOS 全等，5 对编辑参考 latent 全等；无最终 target latent 比对证据。两臂同图、异图各有 7 波真实混进度 DiT B2（[1,0] 至 [7,6]）。开启复用后同图确实仅一次视觉编码，异图同 cohort 两次各自编码。同图 C2 完整 HTTP 末完成 62.0015→58.5157 秒，完成吞吐 0.0322573→0.0341788 图/秒（+5.96%），首张约 56.988→53.451 秒。各格仅两张、一轮；异图到达顺序反转且整体耗时近似，不宣传异图收益。本轮未重测 ComfyUI，没有新的 C1 测量或 P99/SLO 合格结论，首张延迟差距及 L20 仍未解决。实际证据为库外 current-cohort-vision-reuse-ab/{result,independent-analysis,resource-and-png-audit}.json。

该候选完整冻结源码的 Linux check/test/Clippy 全过（顶层 7911 passed、0 failed、81 ignored），正式 CUDA CLI check/build 和正确入口的新版 Rust trace 工具构建通过。Ruby/cc/嵌套 runner 及旧 example 入口失败原件均保留，未改断言或跳过失败。原计划的两张 8MP 普通 enabled CLI 已全 0 退出并恢复 Qwen（14:12:52 UTC）；编辑实际 PNG 为 fa0537e2…，与之前 CPU 噪声诊断保存图全等，root 对照实际原图确认本例未见此前大面积背景波纹，蓝色及原有构图/裂纹保持。生成图唯一原始下载已完成，PNG afcfd68b…已实核；root 目视确认红色茶壶主体和浅色背景成立，未见此前大面积背景波纹，细密釉面纹理及底部小斑点仍可见。两例的语义/视觉验收与 PNG 自比较明确分开；仅此配置/B1/两样本，不扩展到全部输入或 8MP B2。完整普通冷 CLI wall 分别约 106/162 秒，包含加载与退出，不能与 4MP 暖 serve 时间混比。Host0 仍未达到：保留原 16MiB trace 及 304/608B 额外 reservation，不把来源推断当实测所有权；整臂 RSS 采样峰增加约 382MiB，不能声称内存不变。

后续单张扩写优化的 CPU 候选状态算子已通过 active 4 项测试；同计划 1/2/5/1 行算术测试显式预分配编译上限，不等于弹性池碎片重整已解决。全空闲 304/400B 布局不足的原失败和独立压力测试均保留；MTP draft/head、CUDA verifier 和 selected-prefix 事务尚未接入产品，不计作提速。独立测试发现此前 CPU managed I/O 仍只在 active 中，已作为 10d3e65c3 四文件小提交验证并推送（17 项 CPU runtime 测试通过、all-target check/fmt 通过）；候选算子在此前置上隔离验证通过（runtime 1 项、kernel library 377 passed/3 ignored，check/Clippy/fmt 全 0），已以 ee2858e14 提交并推送；先前缺 managed I/O 的 3 项失败保留。新全源码 3729 文件资格快照 c90753cb…仅比原 1d4 增加此 7 文件增量，后续全量回归使用独立远程目录，原实测二进制不改。docs 不暂存、不提交、不推送。

2026-11-07 12:48 UTC 当前状态（优先于以下历史记录）：Qwen-Image-2.1-Turbo 已用 `f4d58c44…` 完成两张生成、两张编辑及一张普通 CLI 编辑；5 个实际 PNG 全等原黄金，CLI/serve 蓝壶编辑的 1,048,576 像素 RGB/alpha 差异全为零，原配置未改，退出与原服务恢复均成功。本轮实际日志只有 B1，不能把 C2 客户端提交称为真实 B2 验收。该二进制不包含后续 H3 视觉复用候选。证据为库外 `current-qwen-shared-regression/{result,independent-analysis}.json`。

同 cohort 精确原图的视觉编码复用已接入显式 `prompt_expansion.vision_reuse`，默认 Off；只复用完成并退休的原视觉输出，每个请求仍独立绑定因果输入，取消来源请求不提前释放幸存者依赖。6 项定向测试、fmt 和 Metal 编译通过，尚无 GPU 性能结果。完整 3,726 文件源码快照已冻结，正在经 Tailscale 中继上传；后续全量 CPU 检查、CUDA 构建及仅 Off/On 两臂 12 图对照按库外 `current-cohort-vision-reuse/ab-preparation/plan-v2.json` 执行。不得将预期节省时间算成已测收益。

独立 F32→BF16 数值合同及 Reference provider 以 `74d47eae4` 提交并推送；隔离工作树 5 项定向测试、受影响全 target 编译及格式检查通过，active 全快照源字节保持不变，docs 未暂存。完整目标审计发现此前 CPU 策略 8MP 编辑验证使用 Disabled 扩写和固定 brief，仍缺当前默认 CPU 策略下两张普通 enabled CLI 的 8MP 生成/编辑；已仅准备这两条必要命令，未执行。此前自然完成/补位/隔离及旧取消证据的版本范围见库外 `current-goal-coverage-audit/gap-audit.json`。ComfyUI 差距仍以 11:50 UTC 实测为准，SLO、性能追平及 L20 仍未完成。

2026-11-07 11:50 UTC 最新里程碑（优先于以下历史记录）：受控 BF16→F16 并行转换核心已以 `63a7e63f8` 独立提交并推送，默认仍为 1 个 worker；产品通过显式 `prompt_expansion.source_conversion_workers=4` 接入 run/serve 共用路径。5090 CPU 资格中 323 个组件 SHA 全等，物化均值 2.621→0.828 秒；完整产品同一新二进制、VAE tile4、4MP/8步、29GiB device/12GiB Host/容量2 的两臂对照共10图，五组 PNG/RGBA、扩写、EOS及参考 latent 全等，两臂各有7波真实混进度B2。单张编辑38.265→36.964秒，双请求末完成65.866→63.845秒，吞吐提高3.17%；这是一轮小样本，相对耗时改善不等于 SLO 达标。原64MiB源缓冲保持，但实际RSS采样峰增加约514MiB，不宣称进程内存不变。

紧接同场 ComfyUI 正常缓存/MTP 配置的5图参考，所有必需节点均实际执行。最新 Ferrum/Comfy 单张编辑36.964/33.217秒（慢11.28%），双请求末完成63.845/58.071秒（Ferrum完成吞吐低9.04%），首张58.770/28.770秒（约2.04倍）。Ferrum真实合批，Comfy这条参考路径并发入队后串行执行；当前最显著差距是首张等待。两边原始输入一致，蓝壶C1/C2的实际扩写文本也一致，绿壶C2扩写不同，不能声称所有下游文本或像素相同。单轮C1一张、C2两张不支持可靠P99/SLO结论。证据在库外 `current-managed-parallel-product-ab/`，本轮没有追平，L20仍不更新。

本地当前源码 fmt/check/test/Clippy及Metal编译全部通过，全量7906通过、0失败、79忽略；首轮7项临时文件写入因磁盘不足失败已保留，清理过期构建缓存后重跑通过。远端正式CUDA check及H3实际构建/出图另行通过。原实验计划的运行中Host0期望未满足：常驻trace占16MiB，空队列时还观察到304/608字节额外committed；两臂相同且C1不累加，源码强指向保留队列容量的reservation，但实机精确sizeof/退出后账本未采集。保留原not_met和偏差说明，不把进程正常退出当作观察到Host账本0。本文仅本地维护，不暂存、提交或推送。

2026-11-07 10:07 UTC 最新验收与性能定位（优先于以下历史记录）：CPU MT19937 image→audio 初始化策略已完成原 8MP 编辑两次复现和三个新 4MP 输入检查，原大面积背景波纹未再出现；这不等于全部提示词保真通过。真实全模型混进度测试已观察 A/B 合批、A 自然完成后 C 补位，B 的 F32、RGBA、PNG 与独立 B1 参考逐字节一致，结束后 Host allocated/committed 均为 0。纯 CPU 噪声模块以 `4689735b7` 单独提交并推送；H3 默认策略已在工作树接入并通过 workspace 与 CUDA 编译检查，本文不暂存或提交。

同一 5090、2048²、8 步、扩写开启的最新阶段对照共 6/6 请求成功；两边各两张预热和一张蓝壶编辑测量。测量图的实际扩写文本同为 918 字节且 SHA 一致。Ferrum/Comfy 完整响应为 40.377/31.895 秒；非重叠阶段分别为扩写 17.931/6.769 秒、参考编码及 conditioning 4.445/4.520 秒、完整采样区域 10.868/16.526 秒、解码 6.020/3.442 秒，其余 1.112/0.639 秒。扩写内的静态装载为 7.567 秒（源物化 4.919、提交等待 2.606 秒）；子项不得重复计入整链，源物化也不等同纯格式转换。参考节点时间是客户端单调时钟接收区间，不是 GPU 核心时间。证据在库外 `current-mainstream-phase-localization/{result,independent-analysis}.json`。本轮仅一个测量样本，尚未建立 P99/SLO 或性能追平；下一步针对扩写重复装载，并用现成 VAE tile1/4 配置作当前版本小对照。此前 R94 arena 分块没有一致整链收益，不推广；L20 更新仍未完成。

2026-11-07 当前状态（优先于下方历史记录）：R91 同一 5090、同一不可变二进制 `697f296b…`、相同 29GiB 总预算与容量 2 的暖态 replay On/Off 对照已完成，8/8 图片及原请求关联、扩写/EOS 摘要、8 步、编辑 latent、容量检查通过；原服务恢复且 controller 正常退出。关闭扩写 decode replay 后，第二次共享 decode 的停顿由 9.778 秒降至 14.828 毫秒；两张 C2 生成端到端时间由 47.266/53.932 秒降至 37.183/43.989 秒，吞吐由 0.037041 提高至 0.045392 图/秒（+22.55%）。只有一轮、每臂两张测量图；两张不同预热图另计，不能作为正式 P99/SLO 或主流追平证明。Off 的 stream 配额 160MiB 仅用于配平原图额度，尚不是最佳产品预设；没有指定具体 CUDA API 根因。证据在库外 `r91-warm-replay-ab/summary.json` 和 `independent-analysis.json`。

R92 同机对照已完成，24/24 输出结构有效；Ferrum/Comfy 吞吐比依次为 C1生成0.681、C1编辑0.673、C2生成0.984、C2编辑0.872。C2首张延迟仍明显落后；每格仅2测量样本、1轮，不能据此宣称P99/SLO或追平。当前697f版本8MP生成/编辑两次CLI均正常退出并恢复原Qwen，实际2912²/8步/B1/预算与轨迹通过，但编辑原图仍有严重背景波纹和过度裂纹，视觉质量明确未通过。R94权重分块候选本地4项定向测试通过，暂停远端构建，优先定位编辑最早差异边界，同时对照Comfy的模型驻留与Ferrum每批重新安装17.9GB权重的路径；不再扩展无整图收益的实验矩阵。资源退休保有提交`85f7b514d`已独立验证并推送；本文不暂存、不提交。主流性能与L20更新仍未完成。

以下为按时间倒序保留的历史实验记录，其中“正在运行”等状态仅描述当时进展。

2026-11-07 R90 封口与回退：5项CUDA编译、4项元数据、7项真实GPU测试和4张实际图均通过正确性，但1280MiB超额限制使两请求37.946/60.675→44.660/61.275秒，吞吐约−1%，总原生分配16.577→17.246秒。原762MB仍miss，慢调用转为246MB约9.73秒及VAE2.416GB约6.93秒，不能因单点变快宣称命中或提速。R89/R90未获益的9源已完整归档库外，并逐SHA恢复active到原R86版本；index/HEAD/远端不可变产物未动，格式和diff检查通过，不把实验能力凑入提交。下一R91已审核授权：原697f/exact缓存，同预算两fresh服务，分别distinct生成/编辑预热后C2生成，共8图，对照replay On/Off，解释已在暖态复现的共享扩写第二波约10.55秒停顿；尚无新结果。24图基线的四格差距仍有效，目标和L20均未完成。

2026-11-07 当前里程碑：R89 同机 24 张对照已完整结束，所有请求、实际 PNG 复读、请求关联与 8 步审计通过，原 Qwen 已恢复。每模式/并发仅 2 样本、1 轮：Ferrum/Comfy 吞吐比 C1生成0.741、C1编辑0.712、C2生成0.803、C2编辑0.861，均未追平；对应端到端P50秒为30.31/22.44、42.76/30.46、52.00/33.80、70.05/47.74。Comfy C2为同时到达后串行执行，Ferrum两模式各7次真实不同进度双成员波；跨引擎前三个原输入的实际扩写文本不同、蓝色编辑相同，不能称下游文本逐字相同。此前R89 unrestricted best-fit的4图正确但吞吐无益：同类759,077,888B尾部分配仍15.36/15.38秒，两个2.034GB缓存块提前被小型长期权重借走；旧762MB过滤遗漏已更正。R90可选最大复用超额限制8文件已应用，本地fmt/check/test/Clippy/Metal全0，7831通过/0失败/79忽略；原预算不变，1280MiB只是实验配置，CUDA资格正在运行，尚无性能结果。C1另有每次17.9GB半精度权重重传6–8秒，暖C2扩写第二波约10.55秒尚无具体CUDA API归因。完整目标、当前8MP验收、SLO及L20均未完成；本文只本地维护，不暂存提交。

2026-11-07 R89 进展：可选最小足够块复用的 9 文件已应用 active；格式、目标 CLI 测试和 workspace check 已通过，全量 tests/Clippy/Metal 仍在执行。R88 两次同机比较均在 Comfy C1 的历史图片复现门槛处终止，尚无完整新基线：首次仅 PNG 提示元数据不同，第二次实际 RGB 最大相差 1 个整数级，约 0.57% 像素不同，扩写文本相同、执行图仅输出文件名不同。两个失败和原图片均保留，原 Qwen 已恢复，未将此判为 Ferrum 回归，也未宣称画质或性能验收通过。已转由唯一远端 owner 执行 R89 独立源码快照的 CUDA 编译与六项实际 GPU 资格验证，保留原二进制；完整主流差距、当前 8MP 验收及 L20 更新仍未完成。本文仅本地维护，不暂存或提交。

2026-11-07 续接：R86 通用原生 backing 缓存已拆为 9 文件提交 `855a502d3` 并 push，默认 Off；独立 HEAD 来源的 CUDA CLI check、lib harness、元数据 1 项、真实 CUDA 3 项通过。首个编译尝试复用了旧 H3 build-script，原失败保留；仅刷新源文件 mtime、全树字节 digest 不变后通过，未补无关 header 或改 feature。完整 active workspace 的既有五项结果仍保持原范围，不把它们冒称为独立闭包的新全量复测。docs 未暂存。R88 同机主流对照已以唯一 controller 3023972 启动（UTC 22:05:21），Comfy C1→Ferrum C1→Ferrum C2→Comfy C2，24 张、相同 distinct 预热和原测量输入、每并发格 fresh 同配置，使用原已验证二进制，不混入新候选。新 R89 可选 best-fit 仍只在库外：命中持完整 native N 的原缓存额度，新的 Core 请求 R 另行计费，避免初版仅保 N−R 时未知释放后缺账；初版已拒绝并保留。修订版仅源码审查通过，尚未应用、编译或上卡，不计作提速。L20 与主流追平验收均未完成。

2026-11-07 R87：同一 R86 二进制、两个 fresh 服务各 2 张 C2 生成，图复用 On/Off 的 4 张实际 PNG、原 golden、扩写/EOS 摘要、8 步及严格轨迹全部通过，均有真实双成员扩写与去噪，原 Qwen 已恢复。Off 用现有 stream quota 字段补齐原图额度，两臂总 opaque/Core storage/private-pool threshold 相同。原 R86 的第二次 decode 约 10 秒停顿在本冷态实验中未复现：On 144.543ms、Off 14.052ms；常态 decode host 累计反为 On 0.576s、Off 2.159s，共享扩写 9.872→10.487s。两张 E2E 为 On 37.877/60.543s、Off 37.496/59.834s，每格仅一轮；不推广关闭图复用，也不与 R86 前序预热后的时间同比。此次两组共同长尾是末单成员 DiT step7 的 762,192,640B 工作缓冲原生分配，分别 15.880737/15.493856 秒，对应步骤 16.715701/16.342463 秒。正在核 exact-plan session 切换及新缓冲分配路径，不能仅凭 resource hash 认定同一物理 pool 实例。新 Nsight 方案仅准备、未执行；下一轮主流引擎同机 C1/C2 对照正在准备。证据：库外 `r87-decode-replay-control/summary.json` 与 `independent-analysis.json`。目标仍未完成，L20 未更新。

2026-11-07 R86：同一不可变 CLI 的显式原生 backing 缓存 On/Off 对照完成，原 controller 2919202 正常退出并恢复 Qwen。16 张 H3 HTTP、1 张独立 CLI 编辑、6 张 Qwen 回归共 23 张输出门控通过；H3 两臂 strict 均 sealed/drop0，16 个请求关联及 8 对同格有效扩写/EOS 摘要/reference latent/PNG-RGBA 一致，两臂各有 14 个真实双成员混进度 wave，容量保持 2、无降级。每格 N1：测量生成 37.1446→27.8960 秒（−24.90%），编辑 40.5016→38.3196 秒（−5.39%）；预热编辑 48.2304→50.5337 秒（+4.78%）。C2 生成整批 54.667→56.227 秒、吞吐下降 2.78%；编辑 71.556→72.255 秒、吞吐下降 0.97%，不能称整体并发提速或 SLO 达标。生成的 76 个 arena 原生分配合计 8.375007→0.035120 秒，两个不同的 2,034,237,440 字节资源后续 10 次均实际命中；每次仍完整上传 323 components/17,907,615,744 字节/503 commands。测量编辑的分配仅减少约 9.5ms，其余收益不能全部归于缓存。固定 4GiB 原 quota 预留、单 entry≤2GiB、最多 16 entries，checkout 保留新的 Core permit/身份与完整初始化，默认 Off；是通用后端能力，不含模型名或资源 hash 特判。实际 native CUDA 3 项与真实 Core 1 项通过，active workspace fmt/check/test/Clippy/Metal 全部通过，7831 顶层测试、0 失败、79 忽略。采样 GPU 峰值 Off/On 为 25062/24902MiB，RSS 约 31.00/31.06GB，均为采样下界；不宣称降低显存。R85 通用 stream/cuBLAS 复用已独立验证并以 `25539a074` push；R86 通用闭包仅准备，未提交。新主流 C1/C2 同机对照已准备，尚未运行；L20 未更新。证据：库外 `r86-native-allocation-reuse-ab/independent-analysis.json`、`resource-analysis.json`、`r86-workspace-validation/summary.json`；本文仍不暂存或提交。

2026-11-07 R85：显式 `backend.native_lane_reuse_maximum_idle_lanes=1` 的原生 stream/cuBLAS 复用候选已完成两项实际 CUDA 生命周期/配额测试、8 张 HTTP 对照与 1 张独立 CLI 编辑，原服务已恢复。相同 5090、4MP/8 步、原权重与容量下，每格 N1：生成 38.6573→37.0620 秒（−4.13%），编辑 51.4046→41.0097 秒（−20.22%）；编辑预热 47.6857→48.2062 秒反慢，不能声称稳定收益、所有请求变快或 C2/SLO 达标。实际复用 22/28 次，新逻辑身份与原 runtime/quota 保持；BLAS 创建 5 次累计 8.651099 秒→2 次累计 0.006897 秒。测量编辑原 8.466835 秒创建停顿被实际复用消除，完整初始化 37.0297→26.6449 秒，未把耗时移到别的测量阶段。8 HTTP 原请求/步骤关联、有效扩写/EOS 摘要、reference latent、PNG 元数据及 sealed strict/drop0 均通过；完整 token ID 未采集；9 张实际文件重读哈希匹配，私有进程正常退出且原服务恢复。候选/基线 1 秒采样显存峰值为 22948/22940 MiB，仅为采样下界。生成仍有 2,034,237,440 字节静态 arena 分配 8.277783 秒，预热编辑仍有 29,511,712 字节动态分配 6.368769 秒。默认仍 Off、L20 未更新。完整 active workspace 五项检查通过，7830 顶层测试、0 失败、79 忽略。通用核心已另拆为基于 HEAD 的五文件候选，并强化单一 owner 的原子唯一性检查；它尚未完成独立 CUDA 验证，不与本轮实测二进制混同，未提交。证据：库外 `r85-native-submission-reuse-ab/independent-analysis.json`、`r85-workspace-validation/summary.json`。

2026-11-07 R84 完成：原R83程序与配置在现32GB WSL重跑4个原输入，测量生成35.9786秒、编辑49.2385秒，4actual PNG、语义与sealed strict均通过，原Qwen恢复，其他15个应用容器未中断。原生vmstat/PSI与API窗口对齐后，4.662s VAE、7.636s大arena、6.688s动态分配、8.797s BLAS长调用期间swap-out均0，swap-in分别约12KiB/8KiB/23MiB/6.2MiB；系统MemAvailable仍约28–29GiB。大量RSS为文件映射，不能据峰值认定匿名内存耗尽；广义phase有压力但不足以解释长停顿，Windows/WDDM未采集。48GB配置只做备份/单行diff及独立Windows进程身份/存活验证，未应用或重启，完整WSL还运行15个非任务应用，当前不为该假说中断它们。R85转为显式typed native stream+cuBLAS owner复用候选，保原quota/失败保有/图与readback退休/新逻辑身份，默认Off；尚无代码实机结果。Spark单次20秒只读连接未完整响应，无状态变更。

2026-11-07 R83 完成：显式256MiB目标分块已通过8张HTTP和1张CLI编辑，实际PNG、扩写/EOS摘要、reference latent与sealed strict均通过；完整workspace五项检查通过，7829顶层测试、0失败、79忽略。实际资源2→76，323权重component/17,907,615,744上传字节不变。测量每格N1：生成36.7473→36.0330秒，编辑50.6606→49.3945秒；暖编辑48.9875→49.7674秒反慢，不能声称稳定收益。生成所有arena分配合计8.0125→8.1948秒，仍被单个2.03GB张量卡8.1947秒；编辑BLAS创建7.9565→8.5764秒。因此没有解决分配/创建停顿，默认不启用此选项。核心与通用配置小提交3639df41b/92fea3f3a已push，诊断程序b70a1d626亦已push，docs未提交。真实5090宿主SEEKEE5090有63.27GiB RAM，但WSL仅30.98GiB；本轮峰RSS28.73GiB、历史swap残留2.56GiB，需用相同二进制验证内存上限假说，当前不足以归因。下一步先备份/核恢复路径，再考虑32→48GiB同输入对照；现未改WSL配置。原Qwen已恢复，GPU/Cargo均空闲，L20未更新。

2026-11-07 R81/R82 最新状态：两轮独立 Rust 显存分配诊断完成，未获得新增整图提速。R81 三种分配接口各一次完整进程均小于1.1秒；R82 在同一个私有池保留2,431,317,344字节存活资源、实际reserved18,656,264,192字节后，最终17.9GB分配及触页同步为29.381ms；先归还空闲页后反而297.330ms（含trim），均没有复现真实模型约7秒停顿。故不更换分配器、不推广主动trim。原Nsight四个5–8秒窗口内已记录kernel/copy/memset/graph重叠均为0，但采集异常结束且缺少UM活动，不能绝对宣称GPU空闲。R83正在既有编译器实现显式typed weight-arena packing选项，初次只接H3扩写，target256MiB、tensor保持连续并允许超target的单tensor独占，默认保留旧布局；等待真实整图A/B。CPU参数/容量测试3项、实际CUDA编译及5个独立GPU场景均通过；初次启动因镜像缺少外部time在0个场景执行前失败，改用Bash内建计时后完成。独立诊断程序提交`b70a1d626`已push，本文仍仅本地。该证据不是模型性能、Comfy等价或正式SLO验收；主流追平与L20部署仍未完成。

R80 最新验收：CUDA正式check/build、3项真实GPU和2项owner测试、四张最小同条件HTTP及完整strict/语义均通过，未推广或提交。相同新服务内gen-measured→edit-measured、每格N1无额外预热：R77→R80生成34.7164→34.7228秒，编辑48.6333→47.6306秒。两臂本轮BLAS构造均不足0.4ms，没有复现原两次生成后出现的7–12秒阻塞，因此不能以该小样本证明R80修复旧性能问题。实际长耗时仍为编辑17.9GB权重arena分配7.375→6.926秒、随后静态加载5.727→5.947秒，以及生成VAE 2.416GB arena分配5.329→4.913秒。下一步检查已有分配器选择/存储复用及参考实现，暂不继续改无关计算算子。Spark一次20秒只读SSH仍在banner阶段超时，旧任务状态未知，无重投；5090原服务恢复正常。

当前继续项：R78 新版Nsight已用同线程的日志→NVTX `cublasCreate_v2`→`cudaMalloc`→`ioctl` 直接定位两次编辑阻塞7.25/8.50秒；四张输出通过，但SIGINT后main栈溢出exit134、严格trace0B，末段有采集丢失警告，不能称完整诊断通过。R79 提前创建计算流句柄虽通过CUDA、8/8 HTTP及完整workspace回归（7820顶层测试），编辑测量54.010→对应基线51.536秒仍回退，已精确撤回active全部4文件，未提交或推广。长构造附近原快照有约16–21GB闲置private-pool缓存，但不是精确调用边界的因果证明。R80复用已有原生库创建前的私有池空闲页回收机制接入cuBLAS，回到R77原初始化顺序、保留live allocation/原quota/错误留存；后续实测结果见上方R80段落。当前已确认的整图收益仍仅R76同会话首编辑减少15.20%，并发与主流引擎追平尚未验收，L20未部署。

2026-11-07 R76/R77 更新：R76 的16 warp F32 causal attention 已通过实际5090长KV/分页/阈值/F64对照，完整4MP编辑的扩写文本、EOS、reference latent及PNG/RGBA与原golden全等。同会话、同配置、新服务首请求（明确cold-declared）R75→R76为53.561466447→45.418127232秒，减少8.143339215秒（15.20%），各1张，仅诊断规模，不代表热态/P99/SLO或主流引擎追平。独立3文件优化提交 `6596f0c24` 已push；此前lane-cap修复 `1758e2994` 已push，本文未暂存。R77 保留F32 structured GQA原adjacent-pair求和树，将warp内归约改为shuffle、减少CTA同步；1个CU+边界fixture已审阅，本地reference及完整workspace fmt/check/test/Clippy/Metal通过（7818顶层通过、0失败、79忽略）。R77 CUDA及2项GPU数值测试通过，On4→R76 On4共8/8图片及完整语义/trace均通过，但measured生成38.439→38.556秒、编辑48.057→50.713秒，未获端到端收益、不推广。扩写视觉4.773→3.114、主视觉3.075→2.270、VAE7.886→6.075秒确实改善；扩写prefill host_encode_submit却1.577→9.119秒（暖编辑同样增加约7秒），内部细项未采集，不能猜测具体CUDA API。下一步用已有新版Nsight复现同HTTP生成→编辑上下文定位主机停顿。R78普通packed decode路由候选及双成员长KV测试仅库外准备，未编译/运行；编辑explicit-RoPE目前逐participant执行，不能把packed路径的潜在收益推广到所有C2。

2026-11-07 最新状态（优先于下方历史记录）：R75 decode replay 已完成实际 CUDA 文本/视觉测试、8/8 HTTP 生成/编辑和独立 CLI 编辑，输出与原 golden 一致，正常退出并恢复 Qwen。相同二进制 Off→On 的测量生成38.802→40.254秒、编辑58.653→56.441秒，各格仅1个样本，方向混合，不支持默认开启或宣布端到端提速；replay 默认仍关闭。完整 workspace fmt/check/test/Clippy 和 Metal 编译均通过，顶层7818 passed、0 failed、79 ignored；CUDA正式功能检查和实际运行另行通过。

新版 Nsight 覆盖完整 CUDA 推理，实有10个 graph各340次复用，共3400次成功 launch。编辑扩写 causal F16 attention 的2728次调用累计10.233秒，每次仅 grid=(1,16,1)、block=(32,1,1)，串行扫描KV，是下一轮优先优化点；累计GPU时间不是请求墙钟时间。R76正在原kernel/launcher内实现多warp分片及F32合并，保持原数值合同、资源接口和共享run/serve路径，先数值检查再完整出图；尚无该候选性能结果。L20仍未部署，正式单图/并发同机追平验收未完成。文档仅本地维护，不暂存或提交。

**状态：目标持续进行；同模型实测仍落后，尚未达到 L20 部署门槛。** 5090 首轮同会话、同官方权重、4MP/8步/真实扩写/C1，预热后文生图 ComfyUI 21.662 秒、Ferrum 43.945 秒，编辑 28.703 秒、63.807 秒。后续同二进制 Fresh/Bounded 对照已完成24/24，测量生成47.184→38.813秒、C2生成吞吐提高6.83%，编辑没有改善；详见下文R72，不能把跨会话数字当隔离因素比较或正式P99/SLO验收。Spark 已测文生图 ComfyUI 37.424 秒、Ferrum 81.316 秒，同样仅单样本；原编辑任务因连接中断尚未取得终态。R71完整workspace矩阵通过；R73解码复用的startup修复已过67项定向CPU测试和workspace fmt/check，完整workspace矩阵亦已通过（7813通过、0失败、79忽略），实际CUDA复用修复资格进行中。8MP 编辑参考画质和完整物理资源资格仍待完成。

2026-11-06 最新硬件安排：用户取消 Qwen 保活要求并要求回归生图主线、提高效率。Spark `172.31.40.42` 经本地及两条中转均未恢复，原编辑任务 controller 55519 状态未知，不重投或盲重启。5090 已通过 Windows Tailscale 中转恢复，候选先无 GPU 编译，再确认 Qwen 空闲并正常停止指定容器开展 H3 对照；保留 Qwen 配置和恢复配方。Qwen 页面原地址为 `http://172.31.40.3:8188/playground`，旧 `192.168.50.173` 已过期。Spark 视频保持停止，embedding 等其他服务保留；两台 GPU 的耗时不交叉比较。

R70 首次 Comfy 请求实际出图，但工作流 FLOAT 字段 `1` 被服务端规范化为 `1.0`，严格图身份审计拒绝。原失败、history 与 PNG 均保留；仅将工作流两个字面量改为 `1.0` 后 R70b 全部通过，没有放宽校验或改模型。R71 在原静态初始化抽象中增加有界来源 range：使用原转换器、原 Host scope 和原 fence，单次安装保留最多64 MiB物理转换缓冲；Fresh 仍为默认。对照借鉴 Comfy 的有界转换/搬运；随后优先复用已有 CUDA Graph 抽象推进扩写 decode replay。Comfy 实际启用动态显存加载和 MTP，不能把差距简单归因于整个扩写模型永久驻留；Ferrum 的图复用还需因果状态、预算与生命周期约束，不能直接打开开关。

2026-11-06 用户更新的交付门槛（优先于此前跨模型 Qwen 相对速度目标）：

- 用主流引擎运行同一 P-H3-Image checkpoint，比较单张延迟与并发请求吞吐。R70b 已在5090建立首个小样本参考，Spark参考亦已完成；下一轮先在恢复连接的5090执行同候选二进制对照。冻结配置、版本、输入、分辨率、步数和精度差异；完整链包含真实扩写，固定同一扩写结果的渲染测量只作诊断。Qwen Turbo与H3的模型结构差异单独解释，不作为同模型验收替代。
- 正式比较前声明 workload 与相对门槛：同条件下 E2E P50/P99 不高于参考，完成图片吞吐不低于参考，零失败，并报告重复、样本量、输出质量和物理VRAM/RSS。并发扫描保持各服务器容量固定，披露 Comfy 单队列/B1 与 Ferrum 实际合批的能力区别，不能将队列并发当合批，也不能故意采用更慢参考配置。小样本诊断不算正式达标。
- 追平后再更新 L20 给用户测试；用户已条件授权此部署。按提供的 `/Users/chejinxuan/ferrum-handoffs/20261106-l20-deployment-guide/Ferrum-L20-deployment.md` 准备 SM89 独立产物、完整native lock、完整模型新OSS前缀、Jenkins固定digest镜像及EAS配置备份/回退。不得把5090 SM120产物直接装到L20，或盲用会覆写启动命令的旧v0.18.3脚本。达标前仅准备，不更新L20。

Qwen 页面地址：http://172.31.40.3:8188/playground ，H3 独占对照窗口会临时正常停止 Qwen，结束按原配方恢复，不保证保活。原 Qwen 二进制SHA `d0d6a61ef7a517ab43a2bc5320ed02179afa907334e23ba1827d5834d8616e9f`。工作分支 `feat/p-h3-image-continuous-batching-20261104`，分块来源接口 `77579181a`、有界归一化转换 `a7233143a` 两个独立提交均已push；本文仅本地维护，不暂存、提交或推送。最新续接记录为库外 `r71-checkpoint.json` 及各硬件证据目录；以下为历史记录，不代表当前服务状态。

2026-11-06 R72 实机结果：同一候选二进制 `4111c039…` 的 Fresh/Bounded 共24/24成功，各自同并发格的 PNG/RGBA、实际扩写 prompt、EOS 摘要和编辑 reference latent 一致；C1另与R70b原golden一致。C1测量文生图47.184→38.813秒（减少17.74%），编辑58.287→61.433秒（增加5.40%）。C2两图批次文生图58.782→55.025秒，吞吐0.034024→0.036347 images/s；编辑116.948→116.772秒，基本无改善，且两个样本完成顺序翻转，不能称所有请求延迟改善。两臂各28次真实双成员混进度前向、0错误/丢轨迹；seed43文生图的扩写在C1/C2间不同，已单独披露，不能宣称跨并发逐字节确定性。每格样本很少、未测物理峰值或正式SLO，Fresh默认不变。完整证据为库外 `r72-r71-source-backing-ab/summary.json`；原controller已exit0且Qwen恢复。初始化各对均改善，但C2编辑一次prefill主机提交调用占17.877/22.321秒，需用现成Nsight定位，不能归因于纯GPU计算或仅靠图复用解决。

下一候选沿原 reusable-execution/cache/容量 owner 接入文本及视觉 decode replay，默认关闭，视觉RoPE/RowSelect仍明确eager边界。原CUDA诊断已确认首次prefill因 `reusable execution preparation is not configured` 返回Interrupted；补回原 `prepare_startup` 后，实机已完成prefill和两个decode，首次复用又暴露admitted provider wrapper漏转发binding-only接口，导致正确裁剪后的调用误入GDN完整编码。v4仅补该wrapper三个转发，未放宽物理view检查、修改精度或owner。原实际CUDA低层缓存测试通过，文本及视觉完整回归等待v4；所有失败原样保留，官方HTTP对照尚未开始。共享代码的startup-v3完整workspace fmt/check/test/Clippy/Metal全部通过：7813顶层测试、0失败、79忽略，另有1条子进程输出不重复计数；v4唯一改动为CUDA文件，Mac不编译该模块，不能据此声称CUDA通过或提速。

库外独立native候选接通typed `prompt_expansion_causal.attention_policy=native-adaptive`，复用现有vLLM V1/V2并保持Portable默认；配置/fixture CPU检查通过，尚未CUDA编译。只读复核发现 pinned native 使用F16 softmax概率、F16 PV单项乘积和V2分区F16中间值，与原Qwen35 F16 profile声明的F32乘法及无分阶段中间舍入不一致，因此暂不推广或按原数值身份运行；需新增准确typed数值profile或更严格native变体。两个接入遗漏（provider旧Portable限制、live native-unit验证）已在独立副本修正。准确的新profile还需closed staged数值合同及native binding贯通，库外设计保留，尚未开始该扩展。

2026-11-06 R73定位进展：5090链路恢复后，原R72二进制进行了C2编辑Nsight诊断，4/4 HTTP成功；SIGINT后主线程栈溢出、exit134、trace文件0B，Qwen已恢复。原报告仍可导出，但Nsight2024.6明确不支持当前driver13.3，CUDA数据只覆盖启动后31.184–65.906秒，目标measured请求在168.068秒才开始，且缺GPU kernel/memcpy活动，因此不能据此解释旧17–22秒停顿或宣称完整GPU时间线。已记录的本次measured分配/同步/BLAS调用没有对应长停顿，但不能回填旧R72。保留原失败，不重投旧profile；核对兼容的官方CLI工具，同时继续旧production单test诊断→startup-v3实际CUDA测试。关闭时栈溢出尚未归因，不能凭0B导出推断内存trace是否已seal。

2026-11-06 R71 更新：核心9文件在 `bb0670922` 的隔离工作树验证，runtime all-target tests为1282 passed、0 failed、6 ignored（不重复计入1个嵌套子进程结果），check/Clippy与workspace fmt均通过。完整H3工作树全矩阵现已通过，证据为库外 `r71-final-workspace-validation/summary.json`；CUDA正确性、Host物理峰值及性能尚待实机，Fresh默认不变。5090 同二进制对照固定 capacity 2、29GiB device/12GiB Host，先 C1 两臂各4张且比对输出，再 C2 各8张，共24张（12预热、12测量）；仍为诊断规模。提交只包含核心接口及其测试，没有提交产品opt-in或本文。

用户追加Spark健壮性：本机为原生NVIDIA DGX Spark Ubuntu/aarch64，非WSL；SSH与NetworkManager开机启用，已补失败5秒重试、SSH协议保活和完整禁休眠。新SSH连接验证通过，原boot与embedding/Comfy进程保持；启动画面等待项新增下次启动90秒超时，当前无限等待没有被中断。未安装Tailscale，未添加认证；通电自启的本机UEFI值仍未获远程读取证据，不能以官方默认值替代实测。Spark软件变更与逐文件回滚记录在库外 `20261106-p-h3-spark/evidence/r71-robustness/summary.json`。Spark同模型Comfy基线已开始，后续与同机Ferrum对照。


2026-11-06 R68 完成 / R69 进行中：

- 原 pedantic F32 SGEMM 不变，逐行 SAXPY 改为一次共享 native 模块广播。4 次捕获的实际 RGBA 与四路 F32 视觉特征逐字节一致，严格 trace 完整；8 次独立计时均关闭 observer。Kernel 模式热态整图 70.443→50.482 秒，与 Off 方向一致。每臂每模式仅一个热态样本，无物理峰值显存/RSS或 SLO 结论，不能把全部收益归于主视觉编码。
- 新全量测试暴露真实取消竞态：旧取消先通知 Condvar，再发资源通知，可能发生在等待方读取状态与进入 wait 之间。修复将队列通知与等待方状态锁配对；确定性旧代码复现 15 秒超时，新代码取消/Drop 边界 2 项及引擎 828 项通过、1 忽略。保留原失败，修复后 workspace 全矩阵进行中。
- R69 仅把该两文件修复同步到已冻结的 R68 共同来源，然后 CUDA check/build 最终 CLI，执行 H3/Qwen 的生成/编辑 C1/C2 共 16 次正确性请求。当前页面保持 R65；尚无新并发耗时或页面推广结论。
- Spark MiniMax-H3 视频服务已恢复并实际产出完整 MP4：256×256、4 秒请求、20 步，实有 107 帧，音视频完整解码通过。页面 `http://172.31.40.42:18190` 已默认使用已验证的 256/4秒/20步；没有据此声称 480/768 或最佳引擎性能已验证。


2026-11-06 r67 完成 / r68 进行中：

- r66 同一官方编辑的四次输出原始像素一致；主视觉 warm 执行 Kernel 9.544 s / Off 9.517 s，affine + patch 复合区间 7.414 s，attention 2.107 s。复合区间包含 enqueue 间隙，不能将 7.414 s 全归为 bias，dispatch 数仅为执行元数据。
- r67 保留原 pedantic F32 SGEMM，将逐行 SAXPY 改为原共享 native module 的一次广播。三项真实 CUDA 数值、runtime 隔离、module/fence/unwind 保有测试均通过；官方 1024 编辑 PNG/RGBA、扩写、EOS 和 reference latent 精确匹配，strict trace observed/sealed、0 dropped。该单次 correctness smoke 的 conditioning_vision 为 3.206856 s，conditioning 总计 5.054251 s，两者嵌套，未据此声称配对提速或 SLO。首次 runner 缺 time 命令及 fixture 非 canonical DeviceId 失败均保留。
- 默认关闭的借用式四特征 observer 与累计 raw payload 有界文件 sink 已通过定向 CPU 测试；本地原冻结源码 workspace check/test 通过，7771 passed、0 failed、79 ignored，Clippy/Metal 仍在进行。新增 profile trace 生命周期补丁另行 CUDA 编译通过，尚未纳入这一批本地全量测试。
- r68 两版程序只在 H3 affine/patch 注册与真实 profile receipt 两个文件不同，共用其余已审计来源。唯一 controller 1512342 执行 4 次 correctness capture 后 8 次 observer 关闭的性能诊断；capture 每臂累计 raw 上限 128 MiB，实际四路特征和 trace 通过才继续计时。页面在独占窗口临时停止，终态恢复 R65，未推广新候选。
- 小提交 228f17c48 已推送：独立 F32 affine 契约 177 行，隔离 runtime 检查/测试/Clippy/fmt 均通过（1274 passed、0 failed、6 ignored）；未提交本文。Spark 已恢复联网，确认旧任务所在 boot 后发生重启且无 MP4；原服务已恢复 ready，单次 256²/4秒/2步功能任务 66335038… 已提交，尚待实际 MP4 验证，不作为质量/性能结果。

2026-11-06 r65 完成 / r66 进行中：

- r65 全工作区 check/test/Clippy/Metal 检查通过，7752 passed、0 failed、79 ignored；3328 个 Rust/native/Cargo 文件前后不变。原 CLI 编辑 PNG/RGBA、扩写、EOS 和参考 latent 均匹配；后续 6/6 HTTP 生成与编辑匹配各自 C1/C2 golden，两组 C2 都有实际混进度 DiT 提交。严格 trace observed、sealed、5738 records、0 dropped。这是正确性回归，没有新速度或 SLO 资格。
- 远端旧 trace consumer 缺少已经存在的 header/producer extensions，首次拒绝 line1。补齐已冻结 consumer 来源后 31 项测试通过，并重读原 CLI 产物通过；未重跑原 GPU 图片，未放宽 schema。CLI 保留原 model identity，仅要求 conditioning_vision；有 service correlation 的 HTTP 编辑另验证 prompt_expansion_vision，未伪造 UUID。
- 试用页已于 06:23:32 UTC 升级至 r65 immutable f5013ec6，保持 Fresh/tile1/capacity2/wait0，LAN 200、ready/idle。暂停期间该页面持续运行。新独立提交 99adbb1d6 已推送，只为既有 Rust profile 示例增加显式配置入口，5 项隔离测试通过；不改变普通 run/serve。
- r66 profiler 构建在暂停期间正常退出0，恢复后核对同一原容器和二进制，没有重复构建。下一步同一官方1024编辑输入、Kernel/Off各一次预热及一次测量，保留完整receipt与像素摘要，只作视觉算子归因。逐行SAXPY优化候选仅在库外准备，尚未应用或验证。
- Spark 较小480预览任务 a3a89f90… 最后确认12/20步骤，之后从Windows与可达Mac mini两条路径均无法访问Spark端点；原因未知、未验证MP4、未重投。库外当前续接点为 r66-checkpoint.json。

2026-11-06 r64/r65 更新：

- 同一 r63 二进制仅切换 Fresh/ReusableSingleSlot，16/16 HTTP 成功，实际 PNG/RGBA、扩写、token/EOS 和参考 latent 均匹配原样本；两份严格 trace 完整。C1 P50 为 65.386→64.090 s，C2 为 116.723→120.087 s，C2 吞吐为 0.016949→0.016480 images/s。初始化虽减少约 4.7 s，并发端到端未获益，未推广复用。每格仅两次测量、固定执行顺序，没有 SLO 或负载显存峰值资格。
- r63 共享 run/serve 的两个实际 CUDA cohort 测试及一次官方 CLI 编辑已通过，CLI 图片、扩写和参考 latent 与 HTTP 对应样本一致。旧 r55f 的 Enabled 4MP generation 混进度、取消和补位证据也经当前严格解析器重新核验；该历史证据不替代当前编辑画质或性能验收。
- 主视觉编码在一次官方 CLI 编辑中执行 10.474 s，占 Conditioning 的 13.334 s 中约 79%；主文本编码执行 0.643 s。下一步优先测量视觉算子开销。F32 affine/patch 的逐行 SAXPY 是源码发现的候选瓶颈，尚未实测其耗时占比。
- r65 仅新增 main conditioning 与 prompt expansion 两种视觉阶段的独立计时，105 项 H3、10 项 recorder、31 项 trace consumer 测试通过。完整 workspace 与 CUDA 实机检查正在进行，不能用 r63 的全量结果代替。页面仍是 r57q2，待官方 CLI 编辑和 HTTP 生成/编辑正确性回归通过后升级，保持 Fresh、tile1 和 capacity2。
- 最近独立提交 `87c6f6d8` 已推送，修复设备不支持 reusable execution 时仍尝试 capture 的运行时行为；仅该切片完成独立红绿验证。本文未暂存、提交或推送。库外当前续接点为 `r65-checkpoint.json`。

2026-11-06 r63 更新：产品视觉 cohort 的 8/8 HTTP 编辑请求成功，实际 PNG/RGBA、扩写摘要、token/EOS 和参考 latent 与 C1 及旧 R60 对应样本一致；两组 C2 各有 333 个双成员 native wave 与 13 个单成员尾波，原请求 frontier 独立。C1/C2 各两次测量中位数为 65.8433/115.9925 s，capture-on 吞吐为 0.0151821/0.0170541 images/s；旧 R60 C2 为 140.1851 s，但跨轮小样本不能作为严格配对或 SLO 通过。新工作区 fmt/check/test/Clippy/Metal 均通过，7748 passed、0 failed、79 ignored。试用页面仍恢复为 r57q2，下一步以同一 R63 二进制仅切换 typed source_backing 做 Fresh/ReusableSingleSlot 对照。Spark 新任务只确认至 3/20 步便再次失联，Windows 和 Mac mini 两条既有中转到 Spark 均超时，未确认 MP4 或 OOM 根因。库外最新续接点为 `r63-checkpoint.json`。

2026-11-06 r61/r62 更新：

- `da1befeb2` 已提交并推送：CUDA 显式声明 fenced static source retention。独立主线代码切片仅改两文件、26 行，CUDA 检查与编译通过，三个实际 GPU 生命周期测试通过；不把该底层改动算作完整 H3 性能优化。
- r62 实际 CUDA 非零视觉 causal 测试通过，包含 13 次安装运行：B1/B2、交换顺序、独立视觉特征与 RoPE 扰动、两成员分别在入队前和视觉 KV 后取消、原执行器退休和 runtime 释放。该 fixture 没有运行真实视觉编码器，不能代替官方编辑全链路验证。证据为 `r62-visual-cohort-cuda/summary.json`。
- 产品状态机复用 run/serve 的 cohort 接口，依次完成每个视觉 producer 的关闭和输入绑定，再让存活请求共享一次 causal 安装；保留原请求身份、内存所有者和失败清理责任。生产源码已冻结，专门的 Host 边界、取消映射与失败持有测试继续进行；官方 1MP 编辑 C1/C2 八请求方案已准备，等待候选构建结果。
- Spark SSH 已恢复，旧 MiniMax-H3 服务进程消失，原视频任务未完成。子代理正在复用已有权重和 SGLang 环境恢复页面与推理，尚不能宣称视频可用。

2026-11-06 r59/r60 更新（库外原始证据位于 `/Users/chejinxuan/ferrum-handoffs/20261104-p-h3/evidence/`）：

- r59/r60 阶段推送 `4604cd3ae`，补齐静态来源安装的计划、Host scope 与真实 retention 校验。邮件修复已合入主线；v0.18.5 的旧发布包没有重新发布。docs 未暂存或推送。
- r58f 完成 32 次 H3 HTTP 请求与 16 次 Qwen 回归，零请求错误；相同并发度的 H3 生成图片一致，Qwen 全部匹配既有图片摘要。实际 trace 确认文本扩写双成员 native wave 和 DiT 不同进度合批，编辑视觉扩写仍逐请求。页面升级至 r57q2；该二进制尚不含 VAE 分块合批。
- r59 显式 `vae_decoder_tile_batch=1..4` 保留默认 1、原精度、256 像素分块及 64 像素重叠。8MP 候选实际执行 60 组、30 个形状 session，仅首次安装权重，后续共享原 owner。生成和编辑的 tile1/tile4 两对 PNG 逐字节相同；两个新增 CUDA fixture 也通过。首次 fixture 因不支持的 Completion 计时模式被拒，改用受支持的 Kernel 诊断后通过，原失败保留，未放宽数值容差。
- 单次 8MP 对照中，生成 decoder 18.389→14.092 s，编辑 18.461→14.344 s；从 trace header 到最后完成阶段，生成 59.574→54.287 s，编辑 124.441→121.702 s。该范围不等于完整进程或 HTTP 耗时，没有重复样本、SLO 或显存峰值资格。原 8MP 编辑纹理问题未因本优化解决。
- r59 冻结生产源码完成 fmt/check/workspace tests/Clippy/Metal 与 CUDA CLI 检查；workspace 为 7731 passed、0 failed、79 ignored，不重复计入子进程测试。后续严格 trace consumer 修复通过 29 项协议测试，并完整读取两份 r58f 和四份 r59 原始 trace；没有丢弃新字段或放宽未知字段检查。
- r60 视觉 cohort 的非零 CPU fixture 验证了 B1/B2、交换成员顺序、单成员视觉和 RoPE 扰动、取消与资源退休。真实 Host 预算现在累计所有成员的持久 Input，精确预算成功、少一字节在上传前拒绝；扩写模块 57 项通过。视觉绑定与 causal 安装已拆开，完整 H3 CPU 范围 94 项通过；这些不是 CUDA 视觉合批或产品入口开放的证明。
- r60 已完成同一 r59 二进制的 tile1/tile4 HTTP 对照：原两条 1024 输入、固定容量 2、C1/C2、每格 1 轮预热和 1 轮测量，共 48/48 请求成功。H3 两臂 16 对 PNG/RGBA 相同，Qwen 16/16 匹配原 golden；实际保存的 48 张 PNG SHA 另行核验通过。生成 C1/C2 中位数为 29.334→29.039 / 34.977→34.361 s，编辑为 68.059→68.590 / 140.185→147.210 s；这组 N=2 对照不支持默认推广 tile4。页面已自动恢复 r57q2。之后新增的视觉 cohort 通用准备接口通过 5 项元数据、9 项真实视觉 CPU、62 项扩写模块测试，服务状态机仍待接入。


2026-11-06 r54 更新：累计 37 个独立代码提交已推送，新增 `777b9dce0`（进程映射权重显式暂存上传策略）、`6b1188d2a`（从冻结 tokenizer.json 补齐特殊词元语义），docs 未暂存、提交或推送。

- r53 完成 workspace fmt/check/test/Clippy/Metal（7621 passed、0 failed、79 ignored）。相同固定容量重复测试中，C1 P50/P99 为 25.013/30.228 s、0.038953 images/s；相比 r51 重复基线 P50 约下降 19.3%。但 C2 为 72.032/83.336 s、0.027318 images/s，较基线回退，不能将 r53 判定为整体成功。12 张 PNG 与 r51 对应样本完全一致。第一次构建进程缓存 27.282 s，计入首轮预热 53.601 s，和 warm 指标分开。
- 两版采样窗口均不完整且延迟超标，benchmark 退出 1；没有完整资源峰值或 SLO 资格。额外关闭资源采样的 C2 仍为 68.528/70.802 s，总 70.820 s；2/2 成功且 PNG 相同，因此慢不能只归于采样。该两请求 read_bytes 增加 36,813,840,384 B，rchar 不变；与映射页读取/工作集压力相容，但不能唯一归因于归一化缓存。
- 闭合 r53 trace sealed、10177 条、零丢失。C2 rep1 比 rep0 多出的约 23.47 s 中，条件编码合计增加 21.17 s；每请求扩写 decode 仍约 5.9 s，去噪未出现同幅增长。静态上传计时包含 runtime submit、分页源处理和 fence，不等于纯 DMA 耗时。
- r54 保留相同归一化映射和精度，新增显式 `upload_policy=staged`，仅去掉该载荷的映射直传授权，复用原有有界 TransferStaging；默认仍为 `read_only_mapping`。两种实际初始化入口及 CLI 配置共 15 项定向测试通过；CUDA 四功能构建、Linux 核心 7 项及正式三功能 check 通过。workspace fmt/check/tests/Clippy/Metal 全部通过（7625 passed、0 failed、79 ignored）。这些检查对应冻结的 r54，不覆盖随后进行的 r55 源码。相同 1024 两输入、固定容量 2、1 预热加 1 测量、无资源采样：C1 P50/P99 32.719/34.264 s、0.030551 images/s；C2 66.364/67.471 s、0.029622 images/s。8/8 请求成功，5090 实际 PNG SHA 与 r51/r53 对应样本完全一致；单图回退，并发仅小样本略低，仍不达 SLO，未接受候选；页面已回退 r51e 且健康。下一轮优先扩写器一次加载服务多请求，再接实际 token 合批与取消隔离。
- Windows 宿主物理 RAM 约 31.72 GiB，WSL 可见约 30.98 GiB，不能靠提高 WSL 上限变出更多物理内存。没有修改宿主配置或重启。Spark 最新只读观察仍超时，视频生成结果未取回。

2026-11-06 r53 进行中：累计 35 个独立代码提交已推送，新增 `116c8fd62`（有界、进程私有的精确执行权重映射核心）。产品接入仍在实机验证；docs 未暂存、提交或推送。

- r52 AVX2/F16C 候选通过全部 BF16 位模式、溢出写入、MXCSR 和 Managed 来源检查，Mac workspace fmt/check/test/Clippy/Metal 也通过。但两条生成 C1 为 31.785/34.769 s、C2 总 68.708 s，编辑 84.179/81.428 s，未显示稳定端到端收益；六张图片像素与 r51 一致。候选保留在库外，工作树和 5090 编译来源已恢复 r51 的转换实现，不将 r52 计作成功优化。
- r53 显式 `prompt_expansion.weight_cache.kind=process_mapped`：第一次实际获准的扩写阶段用原 Host scope 和原转换器逐物理组件生成匿名私有 safetensors，保持必要 F32 参数、F64 affine 舍入及有序拼接；完成后只保留只读文件映射。后续请求复用同一已验证来源、完整物理 schema 的映射，不保留旧 scope、GPU 状态、KV 或提示词。默认 `disabled`，不偷偷降精度或关闭扩写。
- 冷启动仍预留实际来源组及全 schema 流式构建所需的保守峰值；warm 不减少原容量证明。OS page cache/RSS 和磁盘 I/O 是实际成本，不把 mmap 描述为零内存。第一轮仅支持进程私有文件，未实现或声称持久磁盘缓存。run 与 serve 的两个初始化入口共用这项能力，视觉扩写的原 BF16 来源独立保留。
- 新增核心 6 项、模型 3 项、执行入口 1 项、CLI 1 项定向测试通过，核心 Clippy 通过。首轮产品 libcheck 缺错误类型转换、CLI 首轮错误接受 Disabled 的未知字段，均已修正并重跑，失败日志保留。5090 CUDA 四功能构建、Linux 核心 6 项、正式三功能 check 均通过；r53 workspace fmt/check 已过，完整 tests/Clippy/Metal 仍进行中。
- 比较前在库外 `r53-performance-plan.json` 声明相同两条 1024 输入、capacity2、C1/C2、1 轮预热和 2 轮测量、2 秒 Host RSS/整卡显存采样；沿用相对 Qwen 的工程 SLO。r51 基线 12/12 成功，测量每格 n=4：C1 P50/P99=30.983/35.817 s、0.031638 images/s；C2=66.439/69.047 s、0.029583 images/s，均违反延迟目标。所有资源窗口有漏采，最大间隔约 7.7–9.4 s，benchmark 退出 1；聚合峰值不成立，不作 SLO 达标或完整资源资格。r53 候选尚无 GPU 性能结果，首次建文件须与后续请求分开记录。
- Spark 独立 MiniMax-H3 子任务：72.2 GB 权重全部下载并通过 SHA256，SGLang 全组件加载和预热成功；首次 480p/50 步真实视频任务提交后设备失联，未取得 MP4。Windows、Mini、5090 三入口均不可达，原因未确认。子代理已准备更严格内存限制和 TE 映射候选，CPU 测试通过但未部署，不能声称服务可交付；本地只读恢复观察继续运行。

2026-11-05 r51 更新：累计 34 个独立提交已推送，`af56617b4`（初始化内复用已校验的模型/权重计划）以及 `64fe9c6b5`（补推 r48 起已使用且验证的 SSE2/NEON BF16 转换基础，不是本轮额外数值改动）。docs 未提交。

- 原静态初始化对每组来源重复运行完整 schema 校验、JSON 指纹和全来源/逻辑张量遍历。新增私有字段、不可序列化的借用 session，在任何设备写入前校验完整关系一次；每组仍调用原 materializer 的 live descriptor、component/source 和 payload 检查。没有改变量化、预算、分块、fence、源 owner 或核函数。
- 同 5090 相同两输入 1024/C1、全新服务进程、相同 Enabled/max2048/capacity2/29GiB device/12GiB Host：38.936/36.393 s → 33.033/33.747 s，平均延迟降低 11.35%。C2 两请求 57.990/60.199 s，总 60.230 s；编辑 83.254/81.495 s。六张 PNG 与 R50 完全一致。每格两样本一轮，无完整资源采样，不声称 SLO/P99 资格。
- CLI run 成功，PNG 同 HTTP。实际安装 receipt：TE source 2.249 s → 0.008283 s；DiT source 1.701 s → 0.006161 s，DiT total 5.945 s → 2.914 s。复用/关闭重复 receipt 不重复计数。扩写器 source 仍约 7.74 s，转换/分配仍须另优化。
- Rust trace 分析器确认 sealed 4374 条、零丢失；7 次不同 step/timestep 的实际 B2，6 completed、0 failed/cancelled/no-exit。本轮未测取消补位。
- 新增 7 项 session 测试、既有 28 项静态初始化回归均通过；首次新增测试 6 通过/1 失败，原因是 fixture 总 Host capacity=0 不合法，改为 total=1 且四类分配限额均为 0 后重跑 7 通过，原失败保留。CUDA 四功能构建、workspace fmt/check、完整 tests（290 组、7610 passed、0 failed、79 ignored）、Clippy 和 Metal 均通过。Qwen 原配置四格共 32/32 成功，32 张图片与 R48 对应像素摘要完全一致；本轮不采样资源，不作 SLO 资格。正式 CUDA 三功能 check 启动时因本机网络切换 SSH 超时，尚须确认是否创建容器并完成检查。Qwen 首次启动遗漏原模型目录 Docker bind，发请求前失败；补只读 bind 后服务健康，失败日志保留。
- R52 F16C 转换候选在 R51 全量检查完成后已应用两文件；Mac 原 6 项转换测试和 quantization Clippy 通过，x86 交叉编译进行中，实际 x86 硬件测试和性能尚未执行，未提交，不能声称新增收益。Spark 已按用户新要求交由独立子代理部署 MiniMaxAI/MiniMax-H3；原闲置 r48 服务已正常关闭，5090 生图主线继续。

2026-11-05 r50 更新：累计 32 个独立提交已推送，新增 `095cdd760`（通用映射来源与运行时合同）、`77a873c1c`（CUDA/loader 实现）和 `d0339e3a8`（测试输入写法修正）；本 goal 文档继续仅保存在本地。

- 新增独立的只读文件映射来源证明与后端上传能力。普通 stable owner/Vec 不能冒充 mmap；同 scope 的 Managed 路径继续独立。保留原 staging 预留、分块/批次边界、逐来源组 fence 及未知完成状态回收，不把 pageable CUDA 上传宣称为 pinned DMA 或零 RSS。
- 同 5090、相同两条 1024 输入，C1 为 37.886/37.392 s，紧邻旧版为 45.879/44.284 s，平均延迟降低 16.5%。C2 两个请求为 70.661/68.478 s，2/2 HTTP 200；编辑为 87.859/87.259 s，2/2 HTTP 200。六张 PNG 均与上一原生版本逐张相同。每格仅两样本一轮，没有完整资源采样，不构成正式 SLO 资格。
- 实际 CLI `run` 成功，PNG 与 HTTP C1/C2 相同。闭合服务 trace 共 4374 条、sealed、无丢失，Rust 分析确认该 C2 请求对 7 次不同 step/timestep 的实际 B2 前向；本轮未测取消补位。DiT 实际静态加载由旧 CLI receipt 的 11.891 s 降至 5.945 s，13,361,652,224 B 直接使用映射，显式 staging copy 为 0；TE/VAE 有部分等待增加，不能把免复制字节直接换算为全部净收益。
- 针对性测试 20 项通过（8 runtime、3 real archive、2 Comfy carrier、7 原 Managed）；CUDA tensor I/O 测试 6 项通过，四功能构建及正式三功能编译通过，fmt/check 通过。首次新增 runtime 测试 7 通过、1 失败：测试误把独立目标资源的 offset 写为连续地址；只修正测试预期，完整重跑通过。workspace 全量 tests 通过（290 组、7603 passed、79 ignored）。Clippy 首轮拒绝测试中刻意构造的 reversed range 表达式；改为 start/end 输入表，保持同样非法边界覆盖，3 项针对性复测、完整 Clippy、Metal 和 fmt 全部通过。原失败日志保留。
- 已启动同卡 ComfyUI 参考窗口：冻结 56f179 源码、Torch 2.14.1+cu130、comfy-kitchen 0.2.40、相同官方 W4A8/INT8 权重和 BF16 扩写器；同两条 distinct 1024 输入、Enabled max2048/thinking、cache-none。Comfy 的 MTP auto、模型驻留、VAE 默认 F16 和噪声流与 native 有差异，需逐项披露；参考 C1 两请求 29.582/24.885 s；客户端 C2 两请求 47.129/22.455 s，总耗时 47.153 s，4/4 PNG 成功。Comfy 本次是单 worker 队列串行执行，C2 仅指客户端同时提交，不能称为 GPU 合批或连续批处理。首次命令因 closed-loop 误带 maximum-in-flight 在发请求前退出 1；修正命令后成功，旧失败保留。没有资源峰值或正式 SLO 资格。

- 额外 Disabled 扩写诊断：同两输入 C1 为 20.532/7.055 s，2/2 成功；首请求包含初始化，第二请求有缓存复用，不能直接当同分布的冷/热延迟统计。图片与 Enabled 模式不同，默认页面继续使用 Enabled。下一项首要瓶颈是扩写器每请求约 8–9 s 的 BF16→F16 重物化；尚无可直接启用的持久转换缓存，需要原 source identity、转换规则/版本、checksum、原子发布及同一 fence owner。

2026-11-05 r48/r49 更新：已推送 29 个独立代码提交（截至 `1f0302d31`），本 goal 文档仅保存在本地。5090 页面已恢复并运行 r49 候选；尚未完成默认扩写大图连续服务、官方画质与性能资格。

- 同机 Qwen-Image-2.1-Turbo FP8、1024、固定服务容量 2，生成/编辑各 C1/C2，每格 2 个样本、1 轮预热及 3 轮测量：32/32 请求成功，测量每格 6 个请求。生成 C1 P50/P99 为 2.145/2.157 s、0.461 images/s；C2 为 3.315/4.365 s、0.456 images/s。编辑 C1 为 3.220/6.058 s、0.267 images/s；C2 为 4.803/6.383 s、0.313 images/s。全部格子的资源采样不完整，CLI 明确退出 1；不得据此声称完整峰值资源资格或生产 P99。库外记录：`evidence/r48-qwen-fp8-concurrency/`。
- 比较前已在库外 `r48-relative-slo-plan.json` 声明接近 Qwen 的工程目标：相同硬件/工作负载下，P50/P99 不高于 Qwen 的 1.2 倍、有效吞吐不低于 0.8 倍、零请求错误，资源结论须有完整采样。这不是用户明确指定的绝对秒数，也不替代原官方 Comfy 数值/质量基准。当前 H3 明显未达标。
- r49 增加后端声明的 managed static source upload 能力：只直接上传原 Host 账户和原 phase 的真实不可变 WeightBacking；借用不改类别、不新建额度，每个来源组上传完成后才准备下一组。普通 mmap、其他 phase、未支持后端继续使用原 TransferStaging 路径。保留预留不足、未知 fence 与回收边界；新增 7 项硬件无关测试通过。
- 同一 5090、同两个 1024 输入，默认 Enabled 扩写的 C1 生成由 51.485/51.355 s 降至 46.163/42.958 s，两张 PNG 与原生旧版逐字节相同。权重初始化 18.377/20.022 s 降至 12.553/13.506 s，实测 17,907,615,744 bytes 全部复用原 backing，staging copy 为 0。只有 2 个样本、各 1 轮；候选额外编译 FP8 功能但 H3 数值 profile 不变，重启前后冷热状态不同，没有完整资源采样，因此仅为优化诊断，不能替代正式 SLO 对照。
- r49 默认 Enabled 的两请求 C2 生成 2/2 HTTP 200，总耗时 81.659 s、0.02449 images/s，两张 PNG 与 C1 相同。关闭后完整 trace 为 4374 条、sealed、零丢失，Rust 分析器确认该请求对有 7 次不同 step/timestep 的实际 B2 前向；本轮没有测试取消补位。编辑 2/2 HTTP 200，耗时 89.563/92.467 s，其中蓝色编辑与旧版 PNG 相同；CLI `run` 生成成功，PNG 与 HTTP C1/C2 相同。
- r49 fmt/check、workspace tests（289 组、7592 passed、79 ignored）、Clippy 及正式 CUDA 三功能编译均通过，CUDA 上传角色测试 1 项通过，四功能 CLI 构建通过。首次 workspace tests 使用 `--test-threads=1` 时 Criterion benchmark 拒绝参数而退出 2；保留失败记录，改用 `RUST_TEST_THREADS=1` 环境设置后完整重跑通过。Metal all-targets 编译也通过（1m31s）；编译结果不作为 GPU 性能资格。
- r50 优化进行中：实际 CLI receipt 按每次真实安装去重后，图像 TE/connector/DiT/VAE 的显式 staging copy 共 7.664 s；尝试以明确只读文件映射来源证明复用原 owner 上传，保留现有 chunk、额度、fence 和未知提交回收。此数字不是预期净收益，CUDA pageable 上传仍可能内部 staging 或同步，须重新实测。
- Spark r48 默认 Enabled 生图/编辑 3/3 成功；一次只针对本任务闲置模型文件的干净页缓存维护解除了 CUDA 原始 free 观测不足，不是生产准入策略修复。

2026-11-05 r47 更新：用户明确授权按逻辑拆分代码并定期 push；当时已推送 26 个提交至功能分支（截至 `122660e65`），本 goal 文档继续仅保存在本地。准备性源码提交并不代表远端分支已完成产品接线或全目标验收。

- CPU 实机采样定位到每次 DiT 准备及连续补位重复扫描已校验的量化 scale。改为读取原不可变 canonical source 的 metadata，不取消首次物理校验；151 项 H3 CPU 测试通过，5090 18 项 managed 转换测试通过。
- 同 Spark、同 1024、同容量及 MultiShape 配置、Disabled 扩写，r46e→r47b 生成首请求 39.698→26.224 s，后续请求 24.784→11.656 s；编辑 51.928→38.807 s。3/3 HTTP 200，三张 PNG 均与原逐字节相同。每组仅一轮，未测完整资源峰值，不宣称 SLO 或 Qwen 对齐。
- 5090 正式 CUDA 三功能组合构建通过（2m08s）。r47b 页面保持 Enabled/max_new_tokens=2048、29GiB device/12GiB Host、2 槽位，改用 MultiShape4/64MiB，开启不含 prompt 内容的扩写摘要诊断。两个 1024 生成分别为 51.656 s、47.841 s；首张与 r44 原页面 PNG SHA 相同。编辑为 99.161 s，PNG 与 r44 旧页面相同（旧单样本 135.376 s）；旧服务已在无活动请求时正常关闭，保留库外回滚程序。
- r46e 全工作区 fmt/check/test/Clippy/Metal 均通过，串行 test 为 7585 passed、79 ignored；r47b 最终 fmt/check/test/Clippy/Metal 均通过，串行 test 为 7585 passed、79 ignored。Spark r47b CUDA CLI 构建通过（3m29s）。

2026-11-05 r45/r46 优先级更新：用户已试用页面，认为效果接近预期，但等待时间几乎不可用；当前优先优化速度，目标是在同一台 5090、相同分辨率和并发负载下，达到接近 Qwen-Image-2.1-Turbo 的延迟与吞吐。暂不以未经测量的绝对秒数或跨硬件结果宣称达标。5090 页面暂保留原服务，Spark 承担拆因与候选验证。

- Spark 显式关闭扩写、固定 29GiB device / 12GiB Host、2 槽位、1024×1024、SDPA、原 `last_per_stage` 的首轮诊断：生成两次为 51.763 s / 26.595 s，编辑一次为 52.200 s，3/3 HTTP 200。这些是不同冷热状态的单次观测，不是 5090 加速结果或 SLO 通过。
- 原生扩写的有界诊断新增 12 项 CPU 测试已通过；默认不记录扩写正文。H3 接入既有 prepared-family 缓存的改动已完成，正在验证；模型数学、请求身份与容量检查保留。性能收益仍待实测。
- 最新 r43c Qwen HTTP 回归已完成：生成/编辑各 C1/C2，共 16/16 成功，全部 PNG 与上一轮逐张 SHA 相同。当前 1024 同机并发性能基线仍须独立测量。

2026-11-05 r44 已完成状态：

- **先用页面试效果：** 5090 原生 `serve` 已内置并运行 [生图页面](http://172.31.40.3:8188/playground)，5090 局域网地址为 [192.168.50.173:8188/playground](http://192.168.50.173:8188/playground)。当前默认 Enabled 扩写、1024×1024，最大边长 2048、2 个槽位，仍使用原 29GiB device/12GiB Host。真实浏览器生成与继续编辑均 HTTP 200，分别等待 87.15 s、135.376 s，控制台 0 errors；编辑将原红茶壶壶身改蓝，按提示保留红色壶盖/把手与构图。这些仅是单样本端到端功能时延，不是 SLO 证据；收据为 `evidence/r44-playground-browser/result.json`。页面直接调用现有生成/编辑接口，不另建推理路径。
- **官方 4MP 连续服务已取得实证：** r43c 的 5090 官方 2048×2048、Disabled 扩写与 SDPA 测试 1 项通过，总测试时间 185.55 s。实际 12 次提交含 8 次 B2、4 次 B1：先观察 A/B 不同进度同前向，再取消 A、观察 B/C 补位；A 在完成 4 步后于 fence 取消，此后没有 A 的提交，B/C 各完成 8 步。正常 shutdown 后 Host allocated/committed 均为 0，85 条 trace 已 sealed、0 dropped、未截断，容量降级为 0。原图已取回、核 SHA 并查看，证明本次功能与生命周期，不等于参考画质或 SLO 通过；这是实际 `ImageService` 消费方验证，不是 HTTP 传输验收。库外证据为 `evidence/r43c-5090-official-4mp-continuous/` 及对应 `trace-analysis.json`。
- **Qwen 共享路径回归：** Spark Qwen-Image-2.1-Turbo BF16、512×512 实际 B2 生成和 CLI 单图编辑已成功；r43c 实际 B2 测试再次 1 项通过（45.05 s），CLI 构建通过（5m19s）。旧 HTTP 生成/编辑在 C1、C2 四组共 16/16 成功，使用的是既有 r42d binary，不作为 r43c HTTP 资格；新 HTTP 回归因用户优先试页面而暂未运行。生成 seed 43/44 的解码像素差异尚未逐位核验，未建立性能或 SLO 回归结论。
- **已有大图与参考质量边界：** 5090 官方 2912×2912 B1 生成、单图编辑均成功，原图已取回核验并查看。Disabled 扩写、显式 SDPA 的编辑将茶壶由红变蓝，但非目标前景小物件也改变，严格保真未通过；Spark 官方 Enabled 的生成和编辑也已成功。这些运行和页面单次等待均不作为稳态性能或 SLO 证据。库外大图编辑收据为 `r42d-official-edits-image-receipt.json`。
- **官方 Comfy 对照：** Spark 已完成 Enabled/Disabled × 生成/编辑四组 256×256 参考运行，每组 1 请求、1 次重复均成功。库外视觉收据为 `evidence/r43-spark-comfy-reference-visual-receipt.json`。旧原生 Enabled 生成图的茶壶形态弱于参考，未接受为严格画质匹配；native `max_new_tokens=1024` 与 Comfy 生成 `max_length=2048`（编辑为 3072）字段不同。旧 r4 成功日志没有 expanded text、token 数或 finish reason 原记录；当前源码会对 Length 终态拒绝，不能把 cap 差异或“1024 截断”当成已证形态根因。Disabled 编辑的旧原生 seed 42 与参考 seed 43 未对齐。RNG、数值/视觉处理及计时口径的差异仍须披露；有效 PNG 和可见指令跟随不构成质量或 SLO 通过。
- **当前代码验证与剩余项：** r43 analyzer 23 项、engine 756 项（1 项忽略）、HTTP 38 项和 CLI bench 43 项通过；workspace fmt/check 通过。全 workspace 首轮在既有 `cache_recovery` 出现端口竞争，9 项通过、2 项 bind 失败；该组串行重跑 11 项通过（14.44 s），随后完整 workspace 串行测试共 289 组、7560 项通过、0 失败、79 项忽略，原端口 TOCTOU 尚未修复。r44 页面 2 项及完整 server 453 项通过（1.36 s），Clippy（3m24s）、Metal 编译（1m35s）与正式 CUDA 构建（3m01s）通过。默认扩写的正式大图连续加入/取消/补位、参考数值/画质与性能/SLO 仍须分别取得证据，不能由已有功能成功推定。

以下保留此前阶段记录，其中“待验证”等描述对应当时状态。

2026-11-05 05:00 UTC 更新：5090 `run` 使用官方 W4A8/INT8、关闭扩写、2912×2912、seed 42 完成生成，退出 0，PNG 33,952,158 字节，SHA256 `c722b142ea59d8f878ea2fae8c1295aec90c90739f0c6d0a7b06eb0866474806`，完整进程 wall time 954.34 s。日志、退出码、计时和 trace 已收回库外；原图已取回、核对完整 SHA256 并查看，画面为红色茶壶与白色背景，另有提示词未要求的棕色种子；未作系统质量或 SLO 声明。以此原图编辑时，被 17 MiB encoded cap 明确拒绝，未进入 GPU；修复将统一共享 typed 入口上限，仍保留 decoded pixel、codec workspace 和实际 Host 容量检查。r42 暂停恢复 worker 3 项通过（2.40 s），显式 SDPA profile/cache 2 项通过（0.21 s），两种 policy 的真实 refinement Host 查询 2 项通过（1.31 s），CLI policy 2 项通过（0.02 s）；CUDA 全链路尚未执行。Comfy 协议 14 项通过，包含于完整图片基准模块 38 项通过（1.23 s），不等于参考模型性能已测。

r41 已恢复实机验证。2026-11-05，Spark（GB10）通过正式 CLI `run`，使用完整 SHA256 已核验的官方 Comfy W4A8 DiT、W4A8 text encoder 与 INT8 VAE，在禁用扩写、seed 42、B1、256×256 配置下分别完成文生图和单图编辑，每次均有 8 步采样并生成 RGBA8 PNG（各 262,722 字节）。人工查看生成图可见红色陶瓷茶壶与木质桌面；以该图编辑后，茶壶变蓝，木桌及整体构图仍保留。这仅是两个单样本的直观内容检查，不构成参考实现保真或系统质量验收。库外证据为 `evidence/r41-spark-official-256-disabled-r2.{log,png,trace.jsonl,summary.json}`、`evidence/r41-spark-official-edit-256.{log,png,trace.jsonl,summary.json}`，总览为 `evidence/r41-spark-official-256.summary.json`。两个独立 session 的 trace 各含 8 次实际 B1 步骤提交，sequence 1–12 连续、footer sealed、未截断且 dropped=0；生成 shutdown 含 6 份原 session close 和 4 份 shared weights close 凭证，编辑分别为 8 份和 6 份。这些 B1 记录不能证明官方权重的跨请求隔离或连续合批。

生成 receipt 记录的 Host 执行 wall time：TE 77.915 ms、connector 131.532 ms、text refiner 277.341 ms、8 次 steady DiT 合计 2.983431 s、VAE decode 176.208 ms；含 noise 的各阶段执行时间合计 3.647165 s。编辑实际执行 VAE encoder（45.593 ms）、vision encoder（125.513 ms）和视觉条件 TE（106.600 ms），8 次 steady DiT 合计 5.344181 s，所有阶段执行合计 6.086670 s。两份日志均没有端到端计时字段，阶段执行之和不包括完整编译、加载、采样更新、输出与退出开销，不能当作端到端延迟；Peak VRAM/Host memory、吞吐、SLO 和重复测量均未取得。Spark 另有 7 项 CUDA 量化/数值算子测试通过（2.13 s），4 项 tiny H3 CUDA 测试通过（9.55 s），其结果与本次官方权重出图分别记载。

5090 已经由 Mini→Spark→5090 路线恢复连接并同步 r41+p1+p2+p3 源码；三个官方主权重完整 SHA256 与固定清单一致，HF 元数据与 tokenizer 已按原字节同步。原占用服务按授权停止后显存约 442 MiB。编译仍使用 CUDA 12.8，cuDNN 9.20 所需 CUDA 13 动态库以只读挂载补齐。正式 CLI 以 `cuda,vllm-moe-marlin,vllm-paged-attn-v2` 编译成功（3m47s），256 文生图退出 0，图片已查看，与红茶壶提示词相符。Spark 的 1024 文生图也已退出 0 并查看图片，证据为 `r41-spark-official-generate-1024.{log,png,trace.jsonl}`。

5090 固定 B2 服务上的 4MP B1 预热和测量各成功 1 张 PNG、请求错误 0，两张 PNG 的 SHA256 相同。客户端 E2E 分别为 244.534 s、236.427 s；这只是单样本诊断，尚无 SLO 阈值与充分重复。GPU 和 Host RSS 采样峰值分别至少为 29,627,514,880 与 20,764,516,352 字节；每轮各有一次 `nvidia-smi` 查询超时，资源采样不完整，因此 benchmark 退出 1，不能报告完整峰值或性能达标。证据为 `r41-5090-official-4mp-b1.report.json`。后续四请求错峰诊断于 03:36 UTC 观察到 `capacity_reductions=4`、实际 `last_batch_size=1`，故 B2 未达成；已请求正常关闭该诊断服务，保留在途退休和排队退出结果，不能把主动结束后的错误算作完成了四请求验证。

容量瓶颈已定位到 50 层各 6 份 BF16 全行 AdaLN 调制张量：跨层 producer 没有真实 DAG 先后关系，合法内存复用无法消除这 300 份展开，合计每 packed row 3,225,600 字节。r41p6 已增加保持逐次 BF16/F32 舍入的通用紧凑行选择调制/残差算子，替换该展开；CPU 合同、Reference 图与模型回归通过，5090 实际紧凑图对照 1 项通过（1.31 s），完整 tiny H3 CUDA 4 项通过（11.64 s），完整 CUDA 功能 CLI 构建通过（2m52s）。这些小模型结果不替代正式 4MP B2 容量验证。Spark 对应源码 35 文件 SHA 一致，但 tar 保留旧 mtime 导致 Cargo 复用不含新模块的旧 runtime 产物；已刷新受影响源码时间戳重编，不清空缓存或改写指纹。原编译失败证据保留。

5090 原组合测试曾在 3 项通过后段错误（exit 139）。CUDA-GDB 捕获到第二套 pipeline 编辑 encoder 的 `cudnnGetConvolutionForwardAlgorithm_v7` 内部崩溃；loader 日志显示手工加载 cuDNN 组件造成部分卸载、部分驻留后再加载。配置改为仅显式加载 6 个 CUDA 13 依赖，由主 cuDNN 库管理自身组件后，完全相同的原 4 项组合测试通过（12.85 s），随后 r41p6 再次全部通过。未改变 Rust 生命周期、容量账本或添加 NODELETE；vendor 内部引用可使组件驻留至进程退出，不宣称关闭 owner 就卸载了所有库。

Spark 的官方 BF16 扩写器已完整下载并核验固定 LFS SHA256（19,306,305,552 字节）。r41p4 修复同一冻结 tokenizer.json 中的特殊 token ID 解析，4 项新回归和 10 项加载回归通过；r41p5 修正共享 F16 cuBLAS 的 compute type，真实 GPU 4 项测试通过（1.19 s）。默认 Enabled r4 正式 256 文生图退出 0，进程总 wall time 144.97 s（包含初始化和关闭，不等于服务稳态 E2E）；PNG SHA256 为 `0047d90fec71a82e679ef399760f162d6440943126354aea3274ab999b1fb7e4`。已取回查看，画面为红色无盖陶壶与木桌，语义质量仍需官方对照，不能把有效 PNG 当成质量达标。官方独立 Comfy 参考环境已构建就绪，尚未运行对照。4MP B2、8MP B1 生成/编辑、默认扩写编辑、Qwen 实机回归、参考保真与性能验收仍未完成。

r41p7 补齐服务队列中已 Ready、尚未进入 cohort 的请求退出观测：取消/停机记 Cancelled，有限拒绝记 Failed，均先观测原 continuation 再释放原资源，不改变调度或分析器。新增 3 项真实 Stored/native 容量测试通过（2.34 s）；初轮两测试共用 fixture DeviceId 导致 ProcessWide 账本冲突，改为实际独立设备标识后按默认并行重跑通过。直接 `run` 不使用该服务 pending 队列。该修复尚未纳入远端 r41p6 正式 B2 运行。

2026-11-05 04:33 UTC，r41p6 的 5090 正式 4MP 诊断已正常关闭并完整导出：固定 29GiB device/12GiB Host、B2/Auto/maxseqs2/wait0、禁用扩写，四请求按 1000ms 间隔到达。4/4 HTTP 200 和 PNG 完整验证，0 请求错误/拒绝/未发送、0 容量降级；29 条 trace 已 seal、0 dropped、4 个正常完成退出、16 次实际双请求前向。但不同 step/timestep 的前向与 refill 均为 **0**，这轮未覆盖运行中加入，不能称完整连续批处理验收通过。红茶壶样本 PNG 与旧 B1 完全同 SHA256。库外证据：`r41p6-5090-official-4mp-b2.{report.json,trace.jsonl,trace-analysis.json}`。

该次冷启动声明的单轮诊断共 972.731 s，有效吞吐 0.004112 images/s；四请求 E2E 为 483.962/492.595/960.810/969.679 s，P50/P99 为 726.703/969.413 s。未预声明 SLO、未充分重复、没有可用排队/步骤单调时钟数据，文本 TTFT/TPOT/ITL 不适用。GPU 采样 4500 次成功、2 次超时，最大采样间隔 38.699 s；观察峰值 GPU 22,743,613,440 B、Host RSS 20,658,159,616 B 均仅为下界。资源 incomplete 导致 benchmark 退出 1；不报告完整峰值或性能达标，也不将此次冷 B2 与旧 B1 直接作性能比值。

r41p6 Spark 紧凑图 CUDA 对照 1 项通过（1.13 s），完整 tiny H3 CUDA 4 项通过（8.92 s），CLI 重编通过（5m03s）；本地完整 H3 models 192 项通过（9.28 s），服务 steps 26 项通过（6.80 s）。默认扩写编辑随后在正式视觉准备失败：扩写器 hidden1152/16 heads 得到 D72，F32 attention 复用了 BF16 的 power-of-two 限制，CUDA 归约实现也须修正尾部边界。主 TE 视觉是 D64，不能用先前 Disabled 编辑成功替代本次 D72 验证。F32 合同/Reference/CUDA 修复进行中，BF16 旧合同保持。Qwen 正式 B2 GPU fixture 已在 Spark 启动编译，尚无运行结果。

r42 正在实现默认扩写与 DiT 不能同时驻留时的原 runtime 暂停/恢复：保留计费 F32 connector 和原采样状态，真实关闭 DiT owner，原扩写器 Retired 后重装并重新 RefineOnly，禁止旧 cache 跨事务复用。生产 libcheck 修正一个不存在的 allocation API 后通过；真实 Reference native 退休/重装边界 2 项通过（0.64 s），它们不等于完整 H3 CUDA 消费者验证。新完整 CUDA 恢复轨迹、候选取消与最终整树验证仍待完成。8MP 首次启动因容器无 `/usr/bin/time` 退出 127、未执行模型；独立 r2 已改用 Bash 内建计时重新启动，仅验证生成，编辑尚未启动。

r40 已修复连续拒绝候选与取消/补位导致存活请求迟迟不能前向的两条路径：同一组测试修复前 2 项失败、修复后 3 项通过，完整 engine lib 740 项通过、1 项既有诊断忽略。磁盘恢复约 27 GiB 可用空间后，清除了先前 ENOSPC 留下的 5 个不完整链接产物。r41p3 验证命令链中的 fmt、workspace check/test/Clippy、Metal check 均退出 0；由于过程中加入了 r41p4，且随后正在开发紧凑调制算子，该命令链不是最终当前树的统一验收。最终完整检查与后端检查须在新实现稳定后重跑。所有源码、文档与证据均未暂存、提交或推送。

## 1. 目标

在单张 RTX 5090 32GB 上，使 Ferrum 通过原生 Rust/CUDA 执行 **PrunaAI/P-H3-Image 的量化文生图和单图编辑**，同时支持去噪步骤之间的请求加入、退出与补位。多个不同进度的请求能进入同一次真实 DiT 前向，兼顾正确性、容量边界和端到端延迟约束下的吞吐。

已有 Qwen-Image-2.1-Turbo 是现有能力和共享调度回归对象。复用主线的模型执行、资源管理、批处理及基准工具，扩展通用扩散调度，保留两个模型各自的数学语义和数值路线。`ferrum run` 与 `ferrum serve` 使用同一模型执行实现。

首期实施范围建议如下；未纳入的能力不能写入发布支持声明。

| 范围 | 本 goal 的交付边界 |
|---|---|
| 平台与精度 | Linux CUDA / RTX 5090，首期必需路线为官方 Comfy W4A8 DiT、W4A8 text encoder 和 INT8 VAE 组合；保留格式规定的 INT8/BF16/F32 部分。完整 BF16 与全 INT8 产品路线不作为首期必需 |
| 文生图和编辑 | 文本生成单张图片；单张静态 PNG 参考图编辑。覆盖 CLI 与 `/v1/images/generations`、`/v1/images/edits`，保持已有 `n=1` 与 PNG 返回约定 |
| 提示词扩写 | 提供明确的启用/禁用配置，遵循官方默认启用行为；实现受资源预算约束的分阶段驻留。使用原始 prompt 的测量须明确声明，不能与启用扩写的完整链路混比 |
| 连续批处理 | 同尺寸、同模型/精度等兼容域内的不同去噪进度实际合批；不同尺寸分桶调度，避免永久饥饿。首期不要求异形图像共享一次 ragged 前向 |
| 分辨率与容量 | 以 4MP（2048×2048）作为首个实际双请求合批目标；8MP（2912×2912）必须成功完成 B1 文生图和单图编辑，并测明容量。其他尺寸和 8MP 实际批宽按实测声明 |
| 已有能力 | Qwen Turbo 的生成、单图编辑及共享执行/资源边界回归；保留现有 CUDA/Metal 能力，H3 的 Metal 支持不在首期范围 |
| 后续扩展 | H3 的 2–8 张参考图编辑、LoRA、每请求 adapter 切换、多 GPU、视频/音频输出和新量化格式另行定义 |

4MP 下无法实现至少两个请求的真实混合进度前向，意味着首期连续批处理目标未完成，应重新解决容量或执行问题。8MP 只能 B1 时，必须明确其能力是单图执行和并发排队，不能以更小尺寸的结果证明 8MP 连续合批。

## 2. 启动时的基础与差距

最近主线已合入 managed Qwen 生图/编辑（`c05fb98c8`）、并发编辑（`a248683fc`）、计划准备与执行重叠（`14993d727`）及 DiT 共享 FP8 激活量化（`89bf95192`）。这些能力应作为实现基础。

目标启动时，[生图服务说明](image-generation.md)明确在完整批次之间调度。[服务调度器](../crates/ferrum-engine/src/image_generation/service/scheduler.rs)当时一次调用 `worker.execute(prepared, requests)`；[Qwen pipeline](../crates/ferrum-engine/src/image_generation.rs)内部运行完整去噪循环后交还控制权。主线已存在真实批前向与准备流水线，但尚无去噪中途加入；本工作树的步骤调度演进与验证见后续进展记录。

[批准备对象](../crates/ferrum-engine/src/image_generation/prepare.rs)把 prompts、布局、latent 和执行阶段绑定成完整批次。连续批处理需要把请求状态与某次物理批组成分开；实际批宽、mask 和成员变化时，重新校验计划、资源授权与输出归属。

当前原始目录停在较旧的 `perf/prefill-decode-isolation`，包含大量未提交 SLO 实验。本 goal 基于独立主线工作树，沿用[统一执行架构](goal-architecture-refactor.zh.md)；旧状态机不作为本次实现基底。

## 3. 模型与量化路线

### 3.1 固定参考与语义

首轮实现使用以下已核查参考，实际下载仍记录各文件身份；升级版本后重做受影响的数值验证。

| 来源 | 固定 revision / 用途 |
|---|---|
| [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image/tree/7f98acef503cf2bcfba821a78cb1cf1a9928b2df) | `7f98acef503cf2bcfba821a78cb1cf1a9928b2df`，配置与模型语义 |
| [P-H3-Image-Comfy](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543) | `44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543`，官方量化检查点 |
| [Diffusers P-H3 实现](https://github.com/huggingface/diffusers/tree/c55ec27943462eeeb65c177d6ff7d8272d21d7a7/src/diffusers/modular_pipelines/p_h3_image) | `c55ec27943462eeeb65c177d6ff7d8272d21d7a7`，PR #15171 合入版本 |

模型实现需覆盖 `PH3ImageTextConnector`、`MiniMaxH3Transformer3DModel`、`AutoencoderKLPH3Image` 及 `MiniMaxH3Scheduler`。Text connector 消费多层 Qwen3-VL 隐藏态，不能直接替换为现有 Qwen Image 的最后隐藏层输出。VAE、位置编码、条件拼接和输入预处理按 H3 的能力配置独立表达。

保留检查点的 8 步采样网格、时间转换和 `shift_provided_sigmas=false` 等语义；不复用 Qwen 的默认采样参数。官方 [P-H3 pipeline](https://github.com/huggingface/diffusers/blob/c55ec27943462eeeb65c177d6ff7d8272d21d7a7/src/diffusers/modular_pipelines/p_h3_image/modular_pipeline.py)还维护最终丢弃的音频 latent：它参与请求 RNG 消费和联合去噪，须保留等价内部状态及更新，对外仍只输出图片。任何裁剪须先有等价性证据。

### 3.2 5090 量化与资源预算

官方 Comfy 模型卡给出的生图组件数据为：全 INT8 约 30GB 下载、8MP 峰值约 31GB；W4A8 配 INT8 VAE 约 20GB 下载、峰值约 21GB。提示词扩写模型独立计入下载和分阶段驻留预算。**这些是官方单图配置数据，不是 Ferrum 实测容量，也不证明 5090 上 B2 可用。** [来源](https://huggingface.co/PrunaAI/P-H3-Image-Comfy#pick-your-files)

官方 W4A8 使用旋转、码本和分组尺度，并混用敏感层 INT8；现有 Ferrum 的 FP8、GGUF 或 AWQ 路径不能按名称视为等价格式。新增类型化权重描述、metadata 校验、加载/解包与 CUDA 执行，明确旋转、scale、packing、padding、bias 和累加精度。读取 W4 权重后全量展开为 BF16 的参考路径，须如实命名，不能宣称实现了 W4A8 执行。[格式说明](https://huggingface.co/PrunaAI/P-H3-Image-Comfy#pick-your-files)

实际 header 与源码核对还发现 **Comfy checkpoint 使用压缩后的时间条件结构**：`adaln_t_table` 为 F32 `[4097,16]`，时间输入经 clamp、floor 和线性插值形成 16 维 basis；各层 AdaLN 使用 F32 `[96768,16]` 投影与 bias，替代 Diffusers 原始 2688 维时间条件网络。该路径不执行原时间网络的 SiLU，仍保留三种模态与内部音频。加载器必须声明并验证 Dense Diffusers 与 Curve Comfy 两种结构，不能对目标量化 checkpoint 强求不存在的 dense 时间网络权重。Curve 路线的数值基准使用同版 Comfy 逻辑；原始 Diffusers 是独立参考，尚无两者逐位等价或近似误差的实测结论。[Comfy 插值与模型实现](https://github.com/Comfy-Org/ComfyUI/blob/56f179f6c50dad0c4c860ae4da427415c8e86415/comfy/ldm/minimax/model.py#L744)、[由权重形状识别结构](https://github.com/Comfy-Org/ComfyUI/blob/56f179f6c50dad0c4c860ae4da427415c8e86415/comfy/model_detection.py)

容量准入覆盖常驻/暂存权重、扩写模型、条件特征、各请求 latent、attention/算子 workspace、图缓存、VAE decode、PNG 输出及 Host backing。按实际执行计划估算和授权；超预算时有界等待或明确拒绝，不能把触发 OOM 当作容量探索算法。

## 4. 连续批处理执行契约

### 4.1 在既有架构上演进抽象

扩展现有 `ferrum-runtime` / `ferrum-execution` 的 typed tensor 执行和所有权，不新增模型专属运行时或复制第二套容量账本。将当前 `ImageGenerator` 的完整批次调用演进为可逐阶段调度的执行契约；现有只能执行整批的实现应明确报告能力，不能通过外层队列包装宣称支持连续批处理。

下表是拟落地的契约职责，名称可随实际接口审查调整：

| 抽象 | 所有者与不变量 |
|---|---|
| 请求执行状态 | engine/execution 持有稳定请求身份、阶段、输出额度及模型拥有的状态句柄；图像和辅助 latent 的结构由 adapter 定义，不假设只有一张 tensor 或一种 Euler 状态 |
| 模型步骤 adapter | models 描述下一可执行阶段、输入/输出端口、兼容域、计划需求与模型状态更新；Qwen 和 H3 分别实现时间坐标、条件编码、DiT、VAE。scheduler 不解释其数学公式 |
| 批准备与完成凭证 | execution 绑定实际参与者、逐行输入、状态代次、编译计划和资源授权；成员或状态变更后重新校验。提交后凭证按原所有权链完成或隔离，不能重放未知状态 |
| 通用步骤调度 | engine/scheduler 只消费 ready 阶段、兼容键和已证明资源需求，选择实际批次并处理取消、补位与公平性；不出现按模型仓库名选择算法的分支 |
| 设备执行与量化 | runtime 保有唯一资源权威，execution 复用 `TensorComponentSession` / `TensorSerialActor`；quantization 描述存储和数值语义，kernels/native ops 提供匹配的实际运算 |

`run` 通过同一驱动执行 B1，`serve` 驱动多个请求；不各写一套去噪循环。动态成员变化只发生在安全的阶段完成边界，共享单设备执行 lane 仍可完成真正连续批处理，无需为每个请求创建独立 GPU pipeline。先用现有 Qwen 和新增 H3 两个真实消费方验证抽象，不建立无人使用的通用插件框架。

### 4.2 请求与单步批次

每个请求独立拥有 seed/RNG 状态、条件特征、图像及内部辅助 latent、step/sigma、取消状态、输出额度与资源所有者。调度器选择下一个可执行阶段；物理批次只代表这一次前向的参与者。

1. **单步可调度。** 执行接口分为准备、条件编码、一次去噪更新、decode 和退休。一次已提交步骤到达安全完成边界后，调度器可加入、移除或重排成员。
2. **混合进度。** 同一真实 DiT batch 支持每请求独立 timestep、sigma 和更新进度。官方 H3 的复制型 batch 共享部分时间/布局 metadata，需扩展为请求独立表示；分别调用两个 B1 前向不算实际合批。
3. **隔离与身份。** attention 不跨请求，重排/补位保持 prompt、seed、latent、条件和输出的配对。尺寸、精度、adapter、编辑拓扑等兼容规则来自声明能力和计划，不能按模型名称打补丁。只缓存模型语义允许不变的内容。
4. **完成、取消和失败。** 完成请求独立进入 decode/输出；取消请求在当前已提交步骤安全完成后退出后续前向。仅证明未提交的工作可以重试；未知设备状态保留 owner 并隔离，不整批重放或提前复用 backing。
5. **容量与公平性。** pending、准备、active、decode 和输出均有界。新请求的条件编码与扩写也参与同卡调度和驻留预算；不能通过持续准备挤占 active 请求，或用多 pipeline 实例绕开统一资源账本。M0 冻结用于公平性验收的到达负载、调度策略及最大排队/步骤停顿边界，以持续到达测试验证，超出准入能力的请求明确等待或拒绝。
6. **结果可追溯。** 执行记录关联 forward/wave、实际 batch size、各行 request id、step/timestep/sigma 及提交/完成/退休事件。客户端并发量和服务端实际批宽分别报告；不在生产日志记录原始 prompt、图片或整张 tensor。

当前 Qwen 路线每步存在 velocity readback、Host Euler 更新和下一步上传。设备常驻 latent/conditioning、设备端更新及受管理的 gather/scatter 是性能改进候选；需用测量确定收益，不能仅凭代码结构认定主瓶颈。共享调度改造同时验证 Qwen Turbo 的生成和单图编辑。

## 5. 实施分工与里程碑

| 位置 | 主要责任 |
|---|---|
| `ferrum-models` | H3 配置、connector、DiT/VAE、权重映射和模型能力；Qwen adapter 的单步接入 |
| `ferrum-quantization` / `ferrum-kernels` / native ops | Comfy 量化语义与加载、CUDA 运算和必要 native operator 身份 |
| `ferrum-diffusion` | 模型解析后的采样网格、独立请求进度、数值更新 |
| `ferrum-runtime` / `ferrum-execution` | 单步执行授权、状态张量所有权、fence、完成/失败/退休和计划复用 |
| `ferrum-engine` / `ferrum-scheduler` | 请求生命周期、兼容分桶、步骤调度、容量选择和有界输出；沿主线责任边界演进 |
| CLI / server / bench / testkit | 统一配置与产品入口、能力声明、Rust 验证和性能测量 |

| 阶段 | 必须形成的结果 | 退出证据 |
|---|---|---|
| M0 参考与容量准备 | 固定模型/量化/参考实现版本、输入工作集、扩写策略、运行配置，核对 5090 工具链和 native operator lock | 权重/依赖身份、资源预算、同机参考基线；性能候选比较前冻结 SLO 与质量阈值 |
| M1 组件与量化 | connector、关键 attention/DiT block、VAE 和量化算子具备独立数值验证 | Rust fixture 与 5090 真算子对照；混合精度、格式和错误路径有证据 |
| M2 H3 完整单请求 | 同一路线完成 CLI/HTTP 文生图与单图编辑，含配置化扩写和完整采样状态 | 固定输入的数值轨迹/图像质量、协议、容量与资源回收验证 |
| M3 步骤调度 | 请求独立状态、实际混合进度合批、加入/取消/补位、跨桶公平调度；复用已有 Qwen 验证基础 | 可控事件单测及 H3/Qwen 的真实 CUDA 执行记录 |
| M4 容量与性能 | 4MP 实际 B≥2；8MP 能力边界；固定服务容量下并发扫描与优化 | SLO、质量、有效吞吐和完整错误/资源报告 |
| M5 交付检查 | 必需构建/测试、两个产品入口回归、配置示例与准确的支持范围 | 下述完成清单全部满足；记录失败及未验证的组合 |

M1/M2 的 H3 实现与 M3 的通用步骤状态机可以并行推进。不得因已有 Qwen 用例通过就提前宣布 H3 连续批处理完成，也不重新把 Qwen 模型接入列为新功能交付。

## 6. 验证与性能口径

### 6.1 正确性和生命周期

| 场景 | 验收要求 |
|---|---|
| 模型数值 | 分层对照 connector、量化投影、attention、单步输出、采样更新与最终图像；容差按精度在比较前声明。有限值和 PNG 可解码只能证明局部有效性 |
| Late join | A 已去噪后再提交 B；B 在 A 完成前进入同一次实际 DiT forward，记录不同请求进度及正确的逐行参数 |
| 退出补位 | A 完成或取消后退出，B 保持原进度继续，等待的 C 在允许的步骤边界加入；无需等待 B 整个任务结束 |
| 请求隔离 | 同批、重排、取消和槽位重用均不串 prompt/seed/latent；对比单请求参考，按数值容差及图像语义评价，不强求不同 kernel 布局逐像素相同 |
| 真实容量 | 验证合法边界、超限拒绝/等待及释放后重新准入；Host、device、输出和失败 owner 均不超额或提前归还 |
| 失败与背压 | 覆盖准备失败、提交后失败、慢客户端、断连和关闭；不重复推进状态，正常请求与资源最终状态可核验 |
| 产品回归 | H3 与 Qwen 的 `run` / `serve` 覆盖同一模型语义；共享代码变更验证已有 CUDA/Metal 路线，不扩大 H3 后端支持声明 |

复用 [GPU 实批测试](../crates/ferrum-cli/src/commands/image_product/gpu_tests.rs)、[batch 映射测试](../crates/ferrum-engine/src/image_generation/tests/batch_order.rs)及 [HTTP 生命周期 probe](../crates/ferrum-cli/examples/probe_image_lifecycle.rs)。新增测试和可复用基准逻辑使用 Rust。Late join 等时序测试用可控提交/完成事件驱动，不用固定 sleep、机器身份或 PASS 比例判定。官方参考可用于建立数值证据，但不能成为 Ferrum 产品执行的 Python/ComfyUI 代理。

### 6.2 同机测量

性能目标是在**预声明的生图延迟 SLO 和质量门槛内最大化有效 images/s**。首轮报告必须给出数值化的 E2E P99 SLO、测试分辨率/任务类型和适用范围；当前尚未确定阈值，M0 需先测参考基线并冻结阈值，再比较优化候选。缺少阈值或延迟证据时，性能达标状态保持未验证。

使用冻结且可追溯的图像 prompts、编辑输入、seeds、尺寸、步数和扩写策略；所有实现及并发档使用相同样本。先覆盖 4MP、8MP 的生成和单图编辑，混合尺寸/任务到达用于检验公平性。小尺寸与合成输入用于诊断，不能替代正式图像质量和容量验收。

基线使用同一 5090 上固定版本的官方 ComfyUI 量化工作流；Diffusers 固定版本用于数学参考。不同格式、offload、扩写或输出处理差异必须披露。可对齐的测量区间才给性能比值；前端协议不同则分别报告核心推理和完整交付时间，不能混用。图像任务不借用 ShareGPT、llama.cpp 或文本 token 吞吐作为主基线；文本回归继续遵循其既有工作负载要求。

复用 [bench-images](../crates/ferrum-cli/src/commands/bench_images.rs) 和 [图像基准数据结构](../crates/ferrum-bench-core/src/images.rs)。服务容量、批宽上限、精度、模型驻留与配置在一次客户端并发扫描中保持固定；从 C1/C2 起，按实测容量扩展并覆盖饱和点，记录实际物理批宽。分别测冷启动、预热后 closed-loop 和持续到达，不根据结果挑选样本或删除失败请求。

每档报告：有效 images/s、E2E P50/P95/P99、排队延迟 P50/P99、步骤间等待、峰值显存与 Host 内存、请求/图片数、错误/拒绝/未发送到达、重复次数及 SLO 状态。队列和步骤指标需服务端执行事件补充，不能从当前客户端报告推断。文本 TTFT、TPOT、SSE 可见 ITL 对单次 PNG 响应标记为不适用；不得把去噪步耗时改名为 ITL。图像需完整解码并检查尺寸、非空输出、提示词遵循与编辑效果；同样记录异常图和质量退步。

## 7. 完成标准与当前状态

- [ ] H3 原生模型组件、检查点采样语义和官方 W4A8/INT8 路线完成数值与真实 CUDA 验证。
- [ ] `run` 与 `serve` 均能完成文生图、单图编辑；扩写、精度、尺寸/容量限制通过类型化配置、CLI 或文档 preset 暴露。
- [ ] 4MP 实际双请求混合进度前向、late join、退出补位、取消、隔离和故障回收全部有执行证据；8MP B1 文生图及单图编辑成功，实际容量已测明。
- [ ] 共享调度下 Qwen Turbo 回归通过，既有精度、编辑、adapter、prefix 与资源契约无未经说明的退步。
- [ ] 同机正式工作集完成质量、容量和预声明 SLO 下的吞吐验收；缺失项不能以编译或 smoke 通过代替。
- [ ] 稳定代码里程碑通过仓库要求的 fmt、workspace all-targets check/test/clippy；macOS Metal 编译及 5090 CUDA 编译与受影响运行回归有明确结果。
- [ ] 支持文档、示例和 capability 声明与实际验证范围一致；公开提交、PR 和发布按用户指令另行执行。

必需 workspace 检查：

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -A warnings
```

后端编译检查：macOS 上执行 `cargo check --workspace --all-targets --features metal`；配置好 CUDA 工具链及 `FERRUM_NATIVE_OPERATOR_SET_LOCK` 后执行 `cargo check -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2`。编译成功不代替运行、数值或性能验收。

2026-11-04 已通过本地 `ssh-via-tailscale-mini.cjs` 的 `cuda` 路径连通 RTX 5090，读到总显存 32607 MiB、已用 30702 MiB；当时 ComfyUI 与 Qwen Turbo 服务正在运行，Host 内存约 30GiB。此读数只说明当时资源占用，实际测试前重新核对可用资源与运行窗口。连接脚本和凭据、模型权重、参考输出、图片、日志、profile、基准结果及下载缓存留在仓库外。

### 7.1 Spark 与 5090 分工

用户已授权自主使用两台设备。经中转 Mac mini 的 `ssh spark` 确认为 NVIDIA GB10 / ARM64 / SM121，128GB 统一系统内存；本次读到 119GiB 可见总量、85GiB available、约 634GiB 磁盘可用。CUDA 13.0 与 Rust 1.91.0 已有安装；复用其既有 Cargo 缓存，在本任务隔离源码目录编译和测试。

Spark 优先用于 CPU 测试、大容量官方参考和高精度组件验证；5090 保持为 W4A8、真实连续合批及吞吐/延迟验收主设备。Spark 的内存由 CPU/GPU 共享，不能按独立 Host 内存加 128GB 显存重复预算。官方带宽规格为 Spark 273GB/s、5090 1792GB/s；由此不能直接推导实际生图速度比，性能结论仍需各自同机测量。[Spark 规格](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)、[5090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

Ferrum 主线尚无 Linux ARM64 CUDA 发行包；当前 CUTLASS FP8 能力校验只接受 SM89/SM120。Spark 留有旧实验的 SM121 FP8/cuDNN/Qwen 图像执行证据，可供移植审查，但不等于最新主线或 H3 已受支持。复用算子前重新核对源码、构建目标、native lock 与数值能力，不能直接使用 5090 的架构专用二进制。

### 7.2 实施记录

用户再次明确优先使用 Spark 后，Spark 已升级为 H3 CUDA 数值及完整模型验证的主设备，5090 继续承担最终量化容量与性能验收。Spark 原生包兼容问题已通过任务隔离目录中的实际源码构建、依赖锁与 catalog 核对解决；新 native operator lock 的 SHA-256 为 `d2fff16c97046948d24d653db97c3fcb0227d19cc1faec18a65c2fe35458bc17`。没有改写共享旧包、关闭校验或把 SM121 原生包编译结果当作 FP8 数值验证。

- 已完成官方参考版本、当前主线及两台设备盘点；尚未有 H3 完整推理或性能结果。
- 已写入 H3 typed 配置、独立图像/音频采样状态、FP32 connector 参考、布局/RoPE 与 dense/Curve 权重规范；VAE decoder 配置及 Comfy 权重映射独立建模。真实 Stored F32 connector 组件及 Host 适配器已有局部验证；完整 H3 pipeline 尚未注册或跑通。
- Spark 固定源码快照已验证 diffusion 39 项、runtime 862 项（另 5 项忽略）、models 713 项（另 7 项忽略）、Reference 内核 184 项及 Comfy 量化 11 项测试通过；FP32 dense 与 partial RoPE 的真实 Reference 组件测试各 1 项通过。这些是局部正确性证据，不是全 workspace 验证或服务连续批处理完成证据。
- Qwen 已有真实小型 DiT 混合进度、加入/完成补位/重排/边界取消轨迹测试通过，且与独立执行逐步一致。新私有请求状态 owner 已接入 run/serve 共用文生图路径，实际 B1/B2 全轨迹与最后 successor 准入失败原子性两项测试通过；生图执行回归 174 项通过。当前服务队列仍按完整固定批次调度，编辑/prefix 路径尚未迁移。
- 新增真实 Stored F32 connector 及 Host 适配器的 7 项测试通过。r7 固定快照及其最小修正验证 runtime 866 项（另 5 项忽略）、models 720 项（另 7 项忽略）、F32 dense/affine 组件 3 项与 fused RMS/RoPE Stored Reference 组件 1 项通过。
- Spark r7p3 的 7 项真实 CUDA 测试通过：W4A8、W8A8、F32 dense、F32 affine、partial RoPE、fused RMS/partial RoPE、Stored F32 connector。5090 r6p5 的 4 项真实 CUDA 测试通过：W4A8、W8A8、F32 dense、partial RoPE。两组源码范围不同，不能合并解释为同一版本的全模型回归；它们均为小型组件 fixture，不是出图、容量或性能证据。
- r8/r9 新增单次 BF16 舍入 RowRMSNorm、Addcmul 和保留 F32 中间结果的融合 SwiGLU W4A8/W8A8 路线。Spark r9p3 的真实 CUDA 7 项通过（W4/W8、两种融合 SwiGLU、F32 cast/dense/affine），r9p4 的 Stored 单次舍入组件通过。不同数学路径使用独立数值 profile，不把 precise CUDA 算子称作上游 fast-math 二进制逐位等价。
- Spark r9p5 完整 Curve DiT 单层真实 CUDA 测试通过：加载原始 Stored 量化格式，覆盖 AdaLN、QKV、Q/K RMS 与 partial RoPE、分请求注意力、输出投影、MLP 和两次残差；不同去噪进度的两请求具备独立边界/标量 oracle 及请求隔离检查。首次执行准备发现 reshape 输出符号未定，经生产 `compile_options()` 统一声明所有中间形状后通过；没有仅在测试中硬塞形状。该测试使用小型非零权重 fixture，尚未证明官方整模出图、服务端动态入队或性能。
- Spark r9p5 原始 Stored W4/W8 到 F32 激活及输出的两项 CUDA 测试通过；r9p6 runtime 875 项通过（另 5 项忽略）。H3 组件及请求采样测试此前 60 项通过；图像/音频双流采样使用现有 Host 预算，准备全部成功后再提交进度，容量失败或丢弃准备结果不推进请求。当前没有完整 pipeline 消费者。
- Comfy 合并检查点 connector 已支持原生 F32 隐藏态。F32 量化文本编码器单层／多层、GQA、embedding 和隐藏层 taps 已实现，r10 固定快照正在 5090 验证；尚未串通 tokenizer → embedding → 完整文本编码器。初次 CPU 验证发现两处测试代码类型错误，已作最小修正，未放宽数值 oracle。
- VAE decoder 单层和无 RoPE 的 W8 token refiner 已有独立 typed schema/图代码，尚未完成执行验证。VAE 的 F32 FullAttention 是显式 Ferrum profile；官方 Comfy INT8 attention 即使接收 F32 也经过 BF16 输出边界，不能宣称两者数值逐位等价。后续基线必须披露差异。
- 固定 Comfy 源码审计确认：图像节点 shift 为 1/1，参考图像时间下限为 0.999；真实 RoPE 需加载检查点 `inv_freq`；目标采样状态进入 BF16 模型前先舍入，再提升 F32 投影，参考图像保留独立路径；最终调制／投影为 F32，模型输出回 BF16 后才进入对应 velocity 更新。完整组装必须保持这些边界，不能照搬另一路实现的默认值。
- Spark r9p7 的 workspace all-targets check 已通过。此前已补齐 Cargo.lock 对应 registry 缓存及原跟踪的 `docs/release-regression-catalog.json` 编译输入；没有改动该目录文件。完整 workspace test 在既有 ARM `gemm-f16` 汇编中因缺少 fullfp16 编译条件失败；clippy 首次因工具链未安装该组件失败，安装结果待连接恢复核实。正式的全 workspace 与 CUDA/Metal 验收仍未完成。
- r10p1 在 5090 已通过文本编码器 CPU 3 项、F32 runtime 契约 18 项、Comfy 量化 CPU oracle 16 项及 Stored/embedding CPU 集成 2 项；该轮 GPU 验证尚未开始。r11 的通用 VAE 算子、decoder 单层、W8 refiner、多层 DiT 与官方权重单层诊断入口已陆续写入，但未完成统一编译/GPU 验证。多层图复用已验证 block 的 body/schema/shape，层间直接连设备值，生产只保留最终输出。
- r11 初次远端编排只解出了补丁、漏解完整源码包；其成功日志只覆盖旧源码，已标记不可作为 r11 验证。修正同步后，新的 unit RMS/partial RoPE 参数测试因先构造零宽 TensorSpec 而提前失败；仅修正测试操作数，使零属性进入生产几何校验。2026-11-04 本机最小构建已验证该测试及相关 F32 契约共 21 项通过；这不代表新 CUDA 算子运行通过。
- Spark 已完整下载官方固定 revision 的三份量化权重，并按 Hugging Face 原始 LFS SHA-256 校验：DiT 13,362,068,864 字节，文本编码器 3,575,536,021 字节，VAE 3,146,996,440 字节。直连重置后使用镜像作为传输源，身份仍由原始仓库固定 revision 的哈希约束；权重、哈希结果和下载日志全部留在仓库外。5090 权重下载的后续完整性尚未复核。
- 本机剩余磁盘不足以展开新的全量构建，编译/CPU 测试转到 Spark 隔离源码目录并复用既有 Cargo 缓存。代码里程碑仍执行必需验证，不把文档检查替代代码检查。
- SSH 链路有间歇性连接/大流传输超时；已用小分片同步并校验 SHA-256，固定快照后才采信对应测试结果。先前未同步完成的测试日志已标记为旧源码，不作为本次变更验证。现有 GPU 服务未停止。
- 2026-11-04 19:55–19:58（上海）再次按用户要求重试：本机 Tailscale 数据面外网检查正常，但 Mac mini peer ping、Tailscale SSH 22 及脚本既有 LAN 备用路由均超时。远端 GPU 测试和最新小补丁传输因此尚未恢复；继续本地轻量编译和源码实现，未重启 VPN 或既有服务。
- 2026-11-04 20:13–20:14（上海）按用户“重试”再次使用原脚本连接 Mac mini，首跳 SSH 45 秒超时，尚未进入 Spark/5090。没有据此推断设备本身不可用，也没有停止既有服务。
- 本机最小 Cargo 缓存位于仓库外。r12 非 CUDA models all-targets check 通过；H3 首轮测试为 79 通过、1 失败，暴露了通用权重转码掩盖后续层原始 dtype 错误的问题。已在 stack checkpoint 层先验证每层原始 header，原测试不变并复测通过。F32 pointwise 两个 provider 漏声明第二输入的问题已修正，实际 Reference 两项复测通过；通道算子实际 Reference 一项、F32 RMS/RoPE 实际 Reference 四项通过。
- r13 新增最终 F32 调制／投影到正号 BF16 velocity 的真实 Stored family、对应通用行选择调制算子，以及 VAE 单图 Host 布局适配。非 CUDA models all-targets check 通过；H3 与 VAE 的 87 项本地测试全部通过，其中包含 Stored 最后输出层实际 Reference 执行。新算子的契约 1 项和 Reference 内核 2 项通过。该快照没有新 CUDA 结果，不能替代原始整模、连续服务或性能验收。
- 原始发布 DiT header 进一步检查发现：主干 200 个量化投影中实际为 168 个 W4 和 32 个 W8，不能按文件名假定全 W4。旧多层实现复用全 W4 模板，尚不具备完整原始权重加载能力；正在改为根据每个原始投影的量化 metadata 声明并验证格式，禁止用固定层号或模型名选择路线。此前单层 W4 运行证据仍仅覆盖其对应组件。
- r14 已修正混合量化：每个 projection 依据原始 metadata 路由，真实 50 层最小 header fixture 的 1036 个主干物理参数逐一匹配；统一 Stored 的完整 VAE decoder 图也已接入。原始 BF16 小参数仅在实际 Host WeightBacking 授信内提升到 F32，I8 大矩阵保留原格式。首轮编译中两个新测试缺少显式整数／标识符类型，均只补类型标注后复测。
- 2026-11-04 20:46（上海）本机 r14 检查：非 CUDA models all-targets check 通过；models lib **769 通过、7 忽略**；F32 runtime **24 通过**；Reference 内核 **207 通过**；新 Stored LayerNorm 的 Reference 实际执行 **1 通过**。包含共享像素输出适配的 Qwen 回归、H3 混合权重、文本组合、完整 VAE 图与 Host 边界测试；不包含新 CUDA 执行、完整权重出图或 engine 新调度实现。源码快照、清单与日志均在仓库外保存，尚未恢复远端上传。
- VAE 默认路径还需空间分块：固定上游默认 tile=256 像素、最小 overlap=64 像素，先纵向再横向使用已混合邻块融合，最后才做 RGB 反归一化／clamp。整幅图 decoder 与分块路径的注意力／统计范围不同；现已完成图组件不表示默认整图流程完成。单图编辑的 encoder 原始参数为 F32，单帧 temporal kernel 使用最后一个时间切片，空间 reflect padding 和 GroupNorm 也不能替换成现有 Qwen BF16 路径。对应实现继续推进。
- 2026-11-04 21:05（上海）经原脚本重试 Mac mini → Spark，首跳在 SSH banner 阶段约 47 秒超时；Tailscale ping 也无回应。Mini LAN、Spark LAN 和 5090 LAN 的短 TCP 检查未能连通。这里只确认当前连接不可用，没有据此判定 Spark/5090 设备故障；新 CUDA 编译与执行仍未恢复。
- r15 VAE 分块边界 **6 项通过**：精确覆盖画布、纵横交叉接缝、B2／通道隔离、原始 F32 statistics 裁块输入、迟发容量失败后重试同一块，以及最后统一 RGB 归一化。复用已初始化 Stored decoder 的逐块控制器已编译；其真实 CUDA 会话重用／后处理失败保留结果测试尚未执行。
- r16 非 CUDA models all-targets check 通过；完整 DiT 组合 **7 项通过**，包含固定原始 header 的全部 **1074 个物理权重项**、首次／后续图共享声明与 Host 输入／缓存边界，尚无整图 CUDA 结果。编辑 encoder 首段 **3 项通过**，包含真实 Stored Reference 执行与独立 F64 oracle、B2 对 B1 和请求扰动隔离；F32 CNN 契约 **2 项**、Reference 内核 **2 项**通过。这只是完整 CNN 中显式抽取的首段验证，不是完整图片编辑。
- r16 图像 engine 回归 **177 项通过**，包括真实 Stored 小 Qwen DiT 的混合进度加入、取消补位、候选容量不足时旧批次继续，以及最后一个后继分配失败时整批不提交。首轮两项压力夹具未计入分配对齐开销，按真实物理预算修正后原断言全部通过；未放宽 oracle。实际服务端队列接线、H3 pipeline 消费及完整生成仍未完成。
- r17 编辑 Host 边界 **4 项通过**：像素域的独立 F32 舍入、原始 moments 的逐请求拆分、clamp／exp／融合 addcmul／原始 statistics 归一化、容量失败回收。后验输入为显式提供的 F32 噪声，exp 使用声明的 libm profile；没有宣称 Torch CPU seed=42 随机流或 CUDA exp 逐位一致。
- 2026-11-04 21:39（上海）r17 固定里程碑：非 CUDA models all-targets check 通过，models lib **794 通过、7 忽略**，Host memory **37 项通过**（含原子 scope exchange 的 6 项新测试），全 workspace 格式及 diff 检查通过。完整六级 encoder 的 **118 个原始 F32 参数**已接入，真实 Stored Reference 小网络保留全部 12 个残差块并通过 F64／B1／请求隔离验证；这仍不代表官方大权重或 CUDA 编辑成功。DiT 缓存摘要按单请求语义绑定，B2→B1 重组可继续使用自身缓存；不同头布局或归一化参数被拒绝，主干／最后输出时间路由也逐区间核对。
- r17 源码快照在仓库外保存：307 个文件，793309 字节，SHA-256 `3f0bf4ac04fbbd71ea8273551b58964a998fe0b629e0d95a1b6b715ae6fded52`。含明确未注册的 service steps 工作稿；不包含本 goal Markdown，没有暂存／提交／推送。新 CUDA 分支没有编译或运行，当前全 workspace check/test/clippy、Metal、整模 run/serve 与性能验收仍待完成；不得把本机组件通过当成最终支持完成。
- 2026-11-04 22:24（上海）r20 本地里程碑：models lib **806 通过、7 忽略**；连续服务的两项真实小型 Reference 消费测试通过，覆盖混合进度加入、取消补位、实际 VAE 输出与独立 B1 对比，以及 Host 压力下候选等待而已有 DiT 继续推进。服务 steps 已注册到既有 worker；这里仍是 Qwen 消费路径的调度证据，H3 尚未接入完整产品 pipeline。
- H3 新的 RefineOnly 组件与完整 DiT 保持同一物理权重合同；消费者绑定实际 shared-weight receipt，在既有 serial actor 上执行，并先为各请求缓存实分配 Input 背书，再原子发布。速度适配按图像 patch／辅助音频行布局直接准备 F32 successor，包含真实 Reference cast 组件 Output 进入公共采样入口的测试；cast 边界验证不代表完整 H3 DiT 的 CPU/GPU 数值验证。
- 2026-11-04 22:28（上海）HTTP 图片相关 **27 项通过**，包含 F32 RGB 产物编码为真实 PNG 后逐像素读取、输出额度持有与释放，以及原 Qwen BF16 PNG／慢客户端回归。`run`／`serve` 已改用带显式编码合同的同一 owned pixel 产物；CLI 的 GPU feature 编译和运行尚待验证。F32 Philox 契约 CPU 测试通过；新增 CUDA F32 噪声实现未编译或执行，不能宣称与 Torch RNG 等价。
- 21:54（上海）经原 helper 曾短暂得到 `Seekee5090` hostname，但 `nvidia-smi` 不在该 shell PATH；后续连接再度超时，未取得本轮 GPU 状态。22:12 的只读诊断显示 mini discovery pong 可达，WireGuard 数据／TCP22 不通；22:26 明确通过现有 HTTP 代理建立 CONNECT 后，实际 SSH 仍无远端 banner 并超时，SOCKS 代理访问 LAN 设备也失败。此证据不能判定 5090／Spark 本身离线。没有修改原 helper、VPN 配置或现有 GPU 服务。
- 2026-11-04 22:35–22:42（上海）r21：models all-targets check 通过；完整分块编辑 encoder 的真实 Stored Reference 测试通过，保留六级 CNN，按原始相邻 tile 融合 moments，再应用显式 F32 posterior noise；B2 与独立 B1 一致，Host 融合／posterior 容量失败保留实际输出且不重跑设备。`cargo check -p ferrum-cli --all-targets --features metal` 通过，覆盖共享像素产物的 CLI GPU 分支编译；这不是全 workspace Metal 检查或 Metal 运行验证。engine 图片组最终 **181 项全部通过**，新增 native 压力用例通过真实设备账本分配静态 buffer，证明 B2 pool 准入失败后实际回收 journal、原 B1 继续，以及释放外部 owner 后重新加入。下一片 H3 公共 adapter／actor 接线正在编辑，不属于上述里程碑的已验证范围。
- 2026-11-04 23:04–23:45（上海）r23/r24：H3 DiT、VAE 和文本处理公共接口已注册；r24 H3 models 组 **144 通过、0 失败**，models lib 编译通过。文本 BF16 norm 提升及 VAE 最后时间切片打包／BF16 提升使用实际 Host WeightBacking，按实际静态初始化 source group 计算额外峰值；真实 Reference 初始化验证精确额度、少一字节拒绝和释放退款。VAE 资产可为不同编译几何提供同一已验证权重源，缓存对象不持有请求 Host scope。
- 共享 `ImagePipelineExecution` 已成为 runtime、Host、actor、计划、原生权重与回收的唯一执行 owner；H3 通过显式 stage/source adapter 接入，TextRefiner 与 DiT 声明同一静态权重归属。r24 engine 图片组 **187 通过、0 失败**，全 workspace `cargo fmt --all -- --check` 通过。迁移期间 6 个旧 adaptive 夹具仅改阶段 receipt、未改新增声明，现改走真实编译入口，保持全部原有容量／复用断言，并在执行前拒绝身份不一致；相关 13 项先单独通过。新的 shared-weight pin 专项测试尚未注册验证，完整 H3 pipeline 与 run/serve 产品入口仍未完成。
- H3 本地资产元数据入口已校验真实 `modular_model_index.json`、各组件声明、原始配置、tokenizer 与 sigma 网格；请求准备保留各请求独立图像／辅助音频时间基与局部 RoPE 坐标，拒绝混淆请求身份。这里只完成资产、编译配方和执行接口接线，没有完整图片、官方大权重整链路、GPU 连续合批、4MP B2／8MP B1 或 SLO 验收结果。
- 23:12 左右原 Tailscale SSH 路径短暂恢复，实际读取 5090 总显存 **32607MiB、已用 30831MiB**，Spark 可用内存 **87201MiB**。随后整包上传发生 keepalive 超时，后续第一跳 mini TCP22 再度超时，代理 CONNECT 路径亦无 SSH 握手。用户已明确授权夜间停止 5090 占用显存的服务；后续 23:23、23:26、23:29 及约 23:39 的读取均未连通，**尚未执行任何 stop/kill**，没有新的显存释放证据。该授权保留，连接恢复后先核实实际进程并记录恢复方式，再停止占用服务；没有根据旧 PID 操作。
- H3 真实单图编辑还需视觉条件：原 Comfy 路径包括图片标记、F32 视觉计算、三个 DeepStack 残差及多轴位置编码。现有 Qwen BF16 视觉／图像预处理不能直接宣称等价；只提供 VAE 参考 latent 加纯文本也不算完整编辑。视觉元数据解析草稿及本地目录加载草稿暂未注册，不计入本轮验证。所有快照／日志留在库外，goal Markdown 未暂存、提交或推送。
- 2026-11-05 00:05–00:20（上海）r25/r26：官方目录加载接口、视觉元数据解析、共享权重 pin 已注册；目录测试 **6 通过**，视觉元数据测试 **7 通过**，r25 engine 图片组 **195 通过、0 忽略**。这只是加载／声明／共享执行验证，不是完整视觉模型、官方大权重加载或出图。新增保守冷加载预算修复后，实际先安装、准备预算、回收权重、再冷加载的 execution 组 **3 通过**；不能仅因准备时已有驻留权重，就把后续重新安装的临时内存算成 0。
- r26 文本五层输出、标准化、投影与最终 conditioning 的 Host 生命周期查询 **4 通过**；细化阶段按请求独立预分配 cache、实际 work buffer 与迟发失败退款查询 **4 通过**。两组均按原有 Host 的真实 backing 对齐计算，区分各阶段实际存活峰值和一个固定 scope 需要的逐类授信，不建立第二份容量账本。首轮细化 query 的 Rust 可见性错误已修复后复测。
- 同 pin 的 Comfy `sample.py`、`samplers.py`、`nested_tensor.py` 已核实：初始图像／音频只调用一次 `torch.manual_seed(seed)`，随后同一 CPU generator 先按图像 shape、再按音频 shape 执行 F32 randn；不是音频 seed+1，也不是先 flatten 后单次 draw。原生路线选择显式 `NativeCudaPhiloxJointImageAudioV1`：每请求单次 F32 联合 noise 输出，按图像／音频边界分别复制到实际 Input owner，声明与 Torch CPU 随机流不同。新增实现及失败恢复正在本地回归，不计为 GPU 噪声或最终图片验证。
- 本轮不经代理的原 helper 首跳 `100.101.88.54:22`、Mini LAN `172.31.41.153:22`、5090 直连 `172.31.40.3:2222` 均在约 45 秒超时。尚未执行任何停止服务命令，也没有新的显存释放数值；用户“晚上独占、停占用服务”授权仍保留，恢复连接后先读真实进程与启动方式再停止，禁止使用旧 PID。Spark 和 5090 的新 CUDA 编译／整链路验收继续待连接恢复。
- 2026-11-05 00:22–00:32（上海）r27：图片组 **201 通过、0 忽略**，包含真实双 session 失败后保留原始错误／peer／权重 pin／Host scope 并显式回收，以及联合噪声 Output 拆为两个独立 F32 Input owner、后一个分配失败退款与原 readback 重试。噪声测试初轮两项漏计原 Host 为对齐预留的额外 backing，改为核对真实物理 owner 后通过，没有放宽容量边界。阶段编译配方已注册，engine lib check 通过；它仍未接完整生成控制器，不能算作出图成功。
- F32 GELU 的 tanh／erf 独立合同及 Reference 底层 **8 项通过**，随后真实 managed 组件 **2 项通过**，覆盖原 F32 精度、两变体、实际 Output owner 跨 session close 及容量拒绝。CUDA 接线和 ignored 测试已落盘，但尚未编译／执行；不代表视觉整链路已实现。分块 VAE decoder 精确 Host 查询 **2 项通过**，覆盖实际 Host 裁块／频率／unpatchify／混合／canvas owner、对齐与少一字节失败；首次构建磁盘满、测试没有执行，清理任务目录内可重建旧测试程序后重跑通过。所有源码、rlib 缓存、模型与证据保留；新的 CUDA 测试仍未运行。
- r27 注册源码快照在仓库外保存：**417 个文件、1074698 字节**，SHA-256 `71014ed57f9190fe119ac7038c1165030e14ae05b6f4b14d4c4232b7dd1b6120`。明确排除尚未注册的 generation Host 预算与 encoder Host 查询草稿，以及本 goal Markdown；记录 2 个旧像素适配文件的删除。快照未上传，远端仍不能视为最新源码。全 workspace 格式和 diff 检查通过，完整 workspace check/test/clippy、Metal、CUDA 和最终目标验收仍未完成；没有暂存／提交／推送。
- 2026-11-05 00:35–01:00（上海）r28：完整 B1 文生图控制器已注册并编译，实际串接文本五层输出、conditioning、联合噪声、共享权重细化、图像／音频双流逐步推进及分块 decoder；这是源码与编译进展，**尚无完整 GPU 图片**。图片回归 **204 通过、0 忽略**。修复共享 warm session 在旧 scope 关闭后仍使用旧 Output 授信的问题，真实 Stored 重用测试覆盖新 scope 独占额度、输出一致和回收；refiner 也改为显式接收本次 work Host，避免从旧 session 获取已关闭的 Input 授信。执行失败保留原 session、pin、scope 和 backing。
- r28 完整 VAE encoder Host 测试 **2 通过**，包含六层真实 Reference 执行、相邻分块混合、posterior、少一字节拒绝与可重试回收。F32 视觉组件 **3 通过**：三层真实 Stored 图、main 与三个 DeepStack F32 输出对独立 oracle、输入变化和共享权重重绑；F32 patch/reshape 基础契约另 **2 通过**。视觉 CUDA 两算子及完整 ignored fixture 已写入，但没有新 CUDA 编译／执行证据，不能算完整编辑。
- 2026-11-05 01:02–01:15（上海）r29：H3 连续请求状态、一次真实 packed DiT 前向、全批 successor 准备成功后原子推进、原 cohort scope 交换／回滚核心已注册并通过 engine lib check；请求初始化、完成解码及服务接线仍在验证。原生 tiny DiT fixture 复用同一模型 builder 暴露给跨 crate 测试，原始量化 source／共享 schema 检查 **1 通过**；它是小型测试资产，不是官方模型的容量证据。
- r29 模型 all-targets（含 `test-support`）编译通过，视觉与图片预处理组 **6 通过、0 忽略**。新预处理保留 Comfy 的 factor ties-to-even／上下限 resize、忽略 alpha、先 unit F32 RGB 再 bilinear／归一化、两帧重复与 merge-block patch 顺序；独立 F64 oracle 和真实 Host 精确额度／少一字节拒绝测试通过。坐标 F64 与 F32 插值权重作为明确 native profile 披露，不宣称 Torch CPU/GPU 二进制逐位相同。该组件尚未与完整图片标记、F32 TE 视觉注入及多轴位置编码接成编辑入口。
- r28 原 helper 监测的四次第一跳读取均超时；r29 再次移除代理分别走 Mini Tailscale 与 5090 LAN 直连，均约 45 秒超时。**停止服务操作仍为 0**，没有新显存释放或 GPU 性能数值。5090 夜间独占及停占用服务的授权保留；Spark 与 5090 的新源码完整编译／运行验收待可达后执行。所有新日志留在仓库外，未暂存、提交或推送本 goal 文档。
- r29 后续固定点：`generate` 已委托与 `ImageGenerator`／`ImageDenoisingGenerator` 相同的请求、合批与完成消费者，删除旧独立 B1 去噪循环；图片组重新 **204 通过、0 忽略（6.20 秒）**，H3 模型组 **157 通过、0 忽略（18.66 秒）**。CLI、server 沿现工厂增加显式 H3 资产／CUDA 配置分支，初版通过 `--image-config` 选择，未自动推断裸模型目录。CLI CPU all-targets check 通过；随后 CLI 配置测试在构建新 feature 依赖时因磁盘满失败，**测试没有执行**。仅清理该次失败构建新生成的 416180326 字节任务缓存，之前的库缓存、源码和全部证据保留。
- 01:22 左右改用本机 `tailscale nc` 作为原 SSH 的 TCP 传输亦在约 30 秒返回拨号超时，未到远端 shell、未执行停服务。r29 注册源码快照 **457 文件、1170540 字节**，SHA-256 `2ea743dfba4d6ae686a8e266cfd37669eb71d15cb0a221bd839a04378e6eb595`，已验证格式和 diff，未上传。明确排除尚未注册的 Host scope 缩减、服务完成队列及 F32 视觉注入草稿，也排除本 goal Markdown。下一固定点继续修复初始化工作额度保留过久导致不必要拒绝加入，以及完成图片逐个交付时的短时内存背压；GPU 混合进度轨迹和完整编辑仍无验收结果。
- 2026-11-05 01:30–01:57（上海）r30：Host scope 精确缩减与不超过现有空闲额度的缩减已注册，相关缩减 **10 项通过**；它们保持实际 owner 的已用物理字节，避免重新向已关闭 scope 授信。完成队列现先移除并释放已取消请求，再尝试逐张解码／立即交付；全 cohort 共用一个实际阶段额度，初始化暂存和完成请求的音频／cache／权重 pin 在围栏后释放。新增两项真实 Stored DiT/VAE 服务测试 **2 通过（3.02 秒）**，覆盖客户端仍持有首张 canvas 的背压与释放后恢复，以及队首等待时取消后继完成请求让队首恢复。图片执行组 **206 通过、0 忽略（7.45 秒）**；这些是 Reference 小模型服务／资源证据，不是 H3 全模型 GPU 并发出图。
- r30 新增 F32 行替换／加法的运行合同 **2 项**、Reference 底层 **2 项**、实际 managed 组件 **2 项**均通过；视觉主特征与三层 DeepStack 注入采用独立显式 profile，原文本路径保持自己的输入／Stored schema。H3 图片标记、多轴位置编码、实际四路视觉输出到七路 TE 输入的桥接及视觉权重 source 已注册；engine/models all-targets（含 `test-support`）编译通过。随后新增模型测试首次因 fixture 调用了不存在的 `ArchiveTensor::dtype()` 编译失败，改为读取原 safetensors 类型后重跑：图片呈现及真实视觉桥接 **7 通过（0.43 秒）**、量化 TE 视觉图／兼容性 **3 通过（0.04 秒）**、原视觉权重作用域及真实安装边界 **3 通过（1.06 秒）**；H3 模型组 **164 通过、0 忽略（9.16 秒，包含上述图片 7 项，不重复累计）**。CUDA provider／视觉 union 和 preflight 已接线，尚未完成新增 CUDA 编译或执行；完整图片编辑控制器仍在连接，不能宣告编辑可用。此组证据早于下一轮引擎视觉消费者／VAE encoder 执行计数，以及连续前向 helper 抽取和 native decoder 容量调度修改，不能自动覆盖这些后续源码。
- 01:54 左右再次经原 helper 无代理连接，仍在中转 Mini `100.101.88.54:22` 第一跳 **45.017 秒超时**。尚未进入 5090/Spark shell，停服务操作仍为 **0**，没有新的显存释放、完整 H3 图片或性能结果。仅删除本任务已完成测试的可重建程序释放本地链接空间，保留源码、库缓存、日志和模型；CLI 测试此前磁盘满尚未重跑，全 workspace/Metal/CUDA 及最终 4MP B2、8MP B1 验收未完成。本 goal 继续保持未暂存、未提交、未推送。
- 2026-11-05 02:00–02:38（上海）r31：原始 prompt 的文生图与单图编辑现已接入共同的请求准备、初始化、逐步合批、完成和解码控制器。编辑串接独立的视觉／VAE 像素计划、原视觉权重、四路 F32 视觉输出到七路 TE 输入、真实分块 VAE encoder、完整 posterior noise 与参考 latent；参考状态保留原 F32 精度，并进入同一次 packed DiT 前向。严格 PNG 解码在首次设备提交前完成，准备对象绑定原图片 backing，防止换入相同内容但更大父 backing 后沿用旧预算。以上为源码与本地 Reference 组件证据，**尚未执行完整 H3 GPU 编辑或连续生图**；不同请求类型的混合 cohort 及公平性验收仍未完成。
- r31 真实 native decoder 容量测试先出现 **1 通过、1 失败**，定位到 Pool 容量拒绝后没有回收原模型空闲 residency 并重试；生产路径改为先确认 journal 释放，再回收实际空闲 owner，有回收才重试，没有则保留 typed capacity rejection。第二次链接因磁盘满失败、测试未执行；清理本任务可重建缓存后，两项原测试在原容量／断言下通过。像素测试最初 **2 通过、2 失败**，修复已授权工作 scope 内错误地重复申请新 scope 的问题后 **4 通过**。图片执行组最终 **211 通过、0 忽略（7.34 秒）**；H3 模型组 **168 通过、0 忽略（9.47 秒）**；Qwen 图片输入组 **15 通过、0 忽略（4.15 秒）**。前述组件测试已包含在对应组中，不重复累计；Qwen 完整 CUDA 生成／编辑回归尚未完成。
- r31 全 workspace `cargo check --workspace --all-targets` **通过（1 分 38 秒）**，Clippy `-A warnings` **通过（1 分 12 秒）**，`cargo fmt --all -- --check` 与 diff 检查通过；Metal 全 workspace all-targets **编译通过（47.87 秒）**。首次 Clippy 因测试常数触发 `approx_constant`，改为保持相同 F32 bit pattern 的表示后重跑通过。完整 workspace test 仍未完成，本地仅约 394 MiB 可用空间；新 CUDA 编译、GPU 运行、4MP B2／8MP B1 和 SLO 验收均未完成。
- r31 再核目标中的提示词扩写：当前入口仍为 raw prompt，**默认启用的原生扩写尚未实现**，不得视为完整产品链路。已核官方固定 revision 的扩写器 BF16／INT8 ConvRot／W4A8 文件及量化 metadata；现有 Qwen3.5 因果执行可复用，但 Comfy W4/W8 格式、编辑视觉接入和阶段预算仍需开发。不得静默失败回退 raw prompt。02:14 原 helper 无代理连接仍在 Mini 第一跳约 45 秒超时；本机 Tailscale netcheck 的 UDP 正常并不证明 Mini 可达。尚未停服务或取得新 GPU 测量。
- r31 注册源码快照保存在仓库外：**497 个文件、1237330 字节**，SHA-256 `c16a405a0dfbaf1b9b1aa4931c3a3fa963385adaa9981fdf50045d8d484b754a`；已核 tar 清单与 manifest 一致。排除三个尚未注册的 typed edit 服务／CUDA 轨迹测试草稿和本 goal Markdown，记录两个旧像素适配文件的删除。快照未上传；本地源码、goal 与证据没有暂存、提交或推送。
- 02:43（上海）进度复核：r31 快照全部 **497 个文件与当前源码 SHA-256 一致**。再次移除大小写 HTTP／HTTPS／ALL proxy，分别经原 helper 的 Mini Tailscale 中转与 5090 LAN 直连；前者在 `100.101.88.54:22` **45.025 秒超时**，后者在 `172.31.40.3:2222` **45.023 秒超时**，均退出 255。未取得远端 shell，停服务操作仍为 **0**；不能把已有停服务授权报告为已释放显存。目标保持 active。
- 2026-11-05 r32：原视觉图现按声明支持 Qwen3.5 的 main-only 输出，原 Qwen3VL 三路 DeepStack 合同继续严格检查；真实 Stored BF16 视觉权重仍按原数学图执行。视觉组 **19 通过、0 忽略**，包含两种拓扑的实际 Reference 执行、独立 oracle、重绑和回收。主 H3 编辑指令接入官方 `Edit <Picture 1>. ` 呈现规则，原 raw tokenizer 接口保持原义，相关 **2 项通过**。typed edit 服务协议及实际 Stored Qwen 轨迹 **3 项通过**；这不代表 H3 全模型编辑数值或 GPU 资格。
- r32 官方扩写协议已实现并通过 **9 项测试**：原始 Jinja 模板、真实 tokenizer 特殊 token、画幅选择、生成长度、EOS 完成判定及编辑答案格式；没有把协议完成当作原生扩写执行。默认启用／显式关闭的产品配置、真实 causal 推理、编辑视觉注入及阶段内存调度仍未接完，**默认原生扩写仍未实现**。首次测试 fixture 类型编译错误已修正，重跑才计入通过。
- r32 共享 composition 可复用同一 Runtime、provider registry 与 materializers；增加真实静态 Host 初始化入口及原事务回收，**5 项通过**。原 safetensors managed source 按既有 shape 合同支持多源 leading-axis stack／concat，并直接写入一个实际 Host owner，相关 **5 项通过**、既有 archive 回归 **3 项通过**；覆盖真实源顺序、混合精度转换、精确边界及失败退款。静态 Host 测试首次因 fixture 错误未编译，修正后重跑；这些接口尚不包含 causal 动态上传／读回，不等于完整自回归执行。
- r32 原始权重 header 与全部 216 个视觉量化 metadata 已核清：主 TE 的视觉张量在各精度包均为 BF16；扩写器 W4／W8 包中的 108 个视觉投影均为 INT8，qkv／proj／fc1 采用 ConvRot group64，fc2 采用 group16，不能套用现有语言 H256 合同。BF16 扩写器为 775 个 BF16 张量，语言／视觉／非执行 MTP 分区分别为 427／333／15；现有语言前缀和顶层 lm_head alias 可复用。单个 embedding／lm_head 的 BF16→F16 原始权重转换需要约 1.895GiB Host backing，不能只按 192MiB 的合并投影估算。先在 Spark 打通 BF16 路线，5090 上的驻留与量化容量另行实测。
- r32 `cargo fmt --all -- --check`、diff 检查和 workspace all-targets check **通过（编译 46.44 秒）**。完整 workspace test 前两次均在测试开始前因磁盘满而链接失败，第二次已越过 CLI／SLO，失败于 runtime integration target；尚不能报告全量测试通过。仅压缩本任务可重建 Cargo 产物，逐文件核对内容 SHA-256 与 mtime，保留成功链接程序和库缓存；没有清理用户其他缓存、源码、模型或证据。正在据此继续验证，Clippy／Metal 的 r32 结果另记。
- r32 网络复核包括 Mini IPv6、Mini LAN、Spark LAN 直连及原脚本中转，均未取得远端 shell。03:21 左右原 helper 无代理连接在 Mini `100.101.88.54:22` **45.025 秒超时**，退出 255；停服务操作仍为 **0**，没有新增 GPU 显存、完整 H3 出图、实际 4MP B2／8MP B1 或 SLO 结果。所有证据留在库外，goal 文档不暂存、不提交、不推送。
- 扩写精度边界：上述优先 BF16 路线指官方 **BF16 存储权重**。现可复用的 causal 执行是 `qwen3_5.f16`，需真实转换并显式披露 F16 计算，不能称为 BF16 causal 数学或与官方逐 token 等价；若增加 BF16 计算，须独立 profile、provider 和数值资格。当前仅完成 source/header 与基础接口，以上任一路线都尚无完整扩写输出。
- r32 全量测试后续第 3／4／5 次分别在 bench-core／models／execution 的测试程序链接时再次 ENOSPC，退出 101，均尚未开始运行测试。成功构建产物经内容／mtime 校验后透明压缩保留，另删除本任务已完成且与当前 workspace 使用不同 hash 的两个旧 targeted 测试程序；源码、库和证据保留。r32 Clippy／Metal 未运行，不能借 r31 结果自动判定新源码通过。
- r32 注册源码快照：**513 文件、1355451 字节**，SHA-256 `aebe4ec83a328176c0472430c89b39f552c762f7b7c0ada150d76f335de4599e`。逐文件与当前源码核对 SHA，tar 清单与 manifest 一致；排除两个尚未注册的 CUDA 轨迹草稿和本 goal Markdown，保留两个旧文件删除记录。快照未上传。扩写器小型 source/header/metadata 身份证据共 19 项，已在库外持久化并逐项校验；完整 tokenizer 与扩写器权重尚未下载。
- 2026-11-05 03:30–04:10（上海）r33：在原 runtime 上传合同中补齐真实 managed owner 的 participant／offset 范围，ordinary／resident 共用原 owner 和围栏生命周期；相关 **6 项通过**，原 wave identity／legacy 上传回归 **10 项通过**。首次两项测试 fixture 错误已修正，保留原 Step authority 和输入 span 限制。CPU backend 实际受管上传、原 Output 直接读回、边界拒绝与退款 **3 项通过**；首次测试编译 API 错误修正后才计入通过。未增加独立内存账本或绕过原资源 authority。
- r33 原 causal executor 新增明确的 F16／F32 managed greedy I/O profile：上传和读回先在同一实际 Host account／既有 phase scope 预留，再进入原 token 事务。F16 完整 prefill 输出行实际为 `248320 × 2 = 496640B`，F32 才是 `993280B`；decode 为 4B。总预算还包含真实上传及每次分配对齐，不能把 prefill 当作 4B。扫描原输出 owner，沿原 native finite／total-order 规则选 token，保留 signed-zero 与最低索引并列语义。**7 项单元测试通过**；该接口只估计 I/O backing，不代表整个模型或进程的 Host 峰值。
- r33 真实 Stored Qwen3.5 小模型 CPU 联测 **6 项通过（0.67 秒）**：F16 多步对照原 FullLogits 路径、B2 不同 token 行／结果、partial／final prefill 与 decode 混批、实际 F16／F32 精确容量及少 1B 拒绝后同 frontier 重试、策略拒绝与完整退役。执行前查询与实际 executor 查询一致且不授信。首次五项均被 F16 输出与 F32 profile 不匹配正确拒绝；修复为独立 F16 profile 后通过，未改模型数学来迁就接口。这是 causal 运输与生命周期证据，**不是完整扩写器或 H3 GPU 出图**。
- r33 显式单 safetensors 文件 source 已接四种已注册模型 loader，保持 semantic／tokenizer／weights 独立身份及 LFS mmap lease，不扫描邻居替代选定权重。测试 **12 项通过（含 8 项新增、4 项原回归）**，覆盖真实 BF16 条件生成归档的 Qwen3.5 family 准备、顶层 lm_head、非执行视觉／MTP、形状与缺失权重拒绝。首次 fixture enum API 编译错误、第二次错误文案断言失败均保留日志，修正后第三次全通过。
- r33 CUDA Tensor+Causal 共用 Runtime 的显式配置、精确 8 项 dense F16 provider、独立模块额度及 command／fence owner 保留已实现。CPU 可执行的能力／额度政策 **3 项通过**，原 tensor 能力政策 **2 项回归通过**。**CUDA 源码仍未编译／运行，产品构造未接入，模块额度未实测**；stream 数由后续实际 phase 计划声明，不能把这些本地检查当 CUDA 数值或容量资格。
- r33 官方固定 revision 的完整 tokenizer 已下载并校验，小型来源证据增至 **21 项**。实际 tokenizer 的 `<|im_end|>` 是 248046；model／generation 配置的 248044 对应 PAD，官方生成代码明确按 tokenizer 的结束符停止。现有扩写协议应继续沿 tokenizer 身份解析，实际 controller 不得直接沿用 model EOS。完整扩写权重仍未下载。默认原生扩写还缺 managed BF16→F16／DenseAffine 权重 source、phase 内执行与关闭、产品默认配置，以及编辑视觉 token／多轴位置接入；不得静默退回 raw prompt。
- r33 扩写权重预算复核：仅对所选 identity materializer，static 初始化按实际 source group 顺序材料化；单组源转换 backing 与实际 TransferStaging 同时计费，源转换不必形成整份 17.9GB 常驻 Host 镜像。最大 embedding 或 lm_head 各为 2034237440B，不能误算成两个同时驻留，也不能只计 staging。此结论不能扩展到有额外 retention 的其他 materializer。
- r33 新一轮无代理连接：5090 与 Spark 经原 helper 均在 Mini `100.101.88.54:22` 第一跳分别 **45.014／45.022 秒超时**；此前同轮 5090 LAN 直连也约 45 秒超时。未进入远端 shell，停服务仍为 **0**；没有新增 GPU 图片、4MP B2／8MP B1 或 SLO 数据。仅处理本任务可重建缓存以继续本地检查，保留源码、模型、日志和用户其他缓存；本 goal 仍不暂存、不提交、不推送。
- r33 稳定源码检查：fmt／diff 通过，workspace all-targets check **通过（2 分 18 秒）**，Clippy `-A warnings` **通过（2 分 34 秒）**，Metal all-targets **编译通过（1 分 26 秒）**。全 workspace test 首轮在 engine 的 qwen_concurrency 集成测试链接时 ENOSPC，退出 101，测试尚未执行；CLI strip 阶段此前也出现磁盘不足警告。仅压缩本任务已完成编译的 78 个产物，逐文件核对 SHA／mtime 后释放 1031716864B，保留复用缓存，第二轮随后在 runtime 的 vnext_device_operation_wave_execution_contract_tests 链接时再次 ENOSPC，退出 101，仍未开始运行测试。停止构建后仅透明压缩本任务新增的 25 个产物、校验内容与 mtime，释放 700469248B，保留后续复用；全量测试仍未完成。这些编译检查不代替完整测试或 CUDA 运行。
- 04:12–04:17 网络诊断：本机 Tailscale Running、目标路由经运行中的 utun4；Mini 控制面标为 Online，但 Active=false、无当前 endpoint，最近握手仍为昨晚 23:14。随后只针对 Mini 做 1 次 discovery、2 次 TSMP 短探测，均约 3.94 秒超时，未取得直连或 DERP 路径。不能据此判断 GPU 状态；未修改网络或远端服务。
- r33 注册源码快照：**546 文件、1602052 字节**，SHA-256 `777cebd7142c2c97c79a6765d1f3453cd162eeb62fb1edd2d09ceab6bc8b8aa9`。逐文件内容 SHA 与 tar 清单均核对通过；继续排除本 goal 和两个未注册 CUDA 轨迹草稿，记录原两个删除文件。快照留在仓库外、未上传；无大于 1MiB 的新增仓库文件，无暂存或提交。
- 2026-11-05 r34：managed 权重 source 支持原 DenseAffine 的 F64 运算→F32→目标 RNE，保持同 dtype affine 也真实计费，失败回收原 backing。相关 **12 项通过（7 新增、5 原回归）**。模型准备增加绑定原冻结 archive 的长期 mmap 和原 phase Host source；转换预算按实际 identity 静态组峰值计算，**5 项通过（0.21 秒）**。首次模型测试缺 trait import 未编译，补齐原 trait 后通过；未改模型数学或另立内存账本。
- r34 原生文本扩写的借用阶段已串接原注册 Qwen3.5、BF16 存储→F16 计算、实际 Jinja/tokenizer、分块 prefill、greedy decode、tokenizer EOS 和答案提取。**扩写 12 项通过**，其中 3 项为真实 tiny Qwen CPU 模型准备／生成／Length／回收测试，9 项覆盖执行协议、取消、容量及原中断事实保留；首次实际模型 fixture 缺 added_tokens_decoder 导致 3 失败、9 通过，修正 fixture 后全通过。随后接原 retirement custody 的图片执行回归 **226 项通过、0 忽略（7.91 秒，已包含扩写 12 项，不重复累计）**。这些证据不是完整官方扩写器、BF16 计算等价性或 GPU H3 出图。
- r34 原 shutdown Reporter/Observer 新增 `retain_until_retired`，原 phase scope 可转入同一 cleanup custody；取消等待、全部公开 observer 丢弃或 Retained 都不会提前退款，仅实际 Retired 后释放。相关 **12 项通过（7 新增、5 原回归）**，包含真实 Host 授信、注册／退休并发及退休过程中的原错误保留。不能据此宣称已覆盖 initialize 在返回原 owner/cleanup 之前发生 panic 的窗口，产品初始化仍需先建立原 custody。
- r34 默认扩写产品接线尚未完成。现 `start` 同步初始化若直接执行完整扩写，会阻塞原 cohort；后续应让 run/serve 共用可取消、逐步推进的 Pending initialization，在有效扩写文本确定且原 owner Retired 后重新准备 TE/refiner/DiT、缓存和全部预算，同时保留原 caller/PNG 身份。编辑扩写仍缺真实视觉注入与 THW/mRoPE；不能静默跳过扩写或把 main TE 的 DeepStack 当作扩写器输入。CUDA Tensor+Causal union、实际 stream/模块预算和原静态初始化的 panic custody 尚未接完整。
- r34 04:54–04:55（上海）无代理复测：5090 与 Spark 经原 helper 均卡在 Mini `100.101.88.54:22` 第一跳，约 45 秒后退出 255；三个已知 LAN 端口的一轮 TCP 探测均未建立连接。Tailscale 控制面 Online 和本机 netcheck 正常不证明 Mini SSH 可达。未取得新 GPU 状态，停服务操作仍为 **0**；无完整 H3 图片、4MP B2／8MP B1 或 SLO 结果。证据为库外 `hardware-r34-20261104T205427Z-summary.json`。
- r34 注册源码快照共 **561 文件、1647519 字节**，SHA-256 `e53be803271c17e694e32c40f5eab1cc907ff313b8afbb9e6e0e0bfdfe565621`；逐文件内容 SHA 和 tar 清单一致。继续排除本 goal 与两个未注册 CUDA 轨迹草稿，保留两个删除记录；快照仅存库外、未上传。为后续检查，仅透明压缩本任务 13 个 Cargo 产物，逐文件核对 SHA/mtime，释放 304164864B，保留成功链接程序；未处理用户其他缓存或模型。没有暂存、提交或推送。
- r34 稳定源码检查：最终 fmt／diff 通过，workspace all-targets check **通过（1 分 28 秒）**，Clippy `-A warnings` **通过（1 分 35 秒）**，Metal all-targets **编译通过（1 分 24 秒）**。本轮未重跑全量 workspace test：此前 r33 两次均在测试执行前链接 ENOSPC，本轮检查后可用空间约 478MiB；全量测试仍未完成。CUDA 编译和 GPU 执行因无可达 CUDA 主机未做，不能用 CPU／Metal 编译结果替代。

- 2026-11-05 r35：在原 `CompiledProgram` 构造、原 ExecutorOwner/cleanup 与 Host account 中接入 owned phase custody，未新建并行资源账本。构造边界 **5 项通过**，覆盖成功、实际转换额度少 1B、未知围栏保留、原始 panic 传播；native acquisition 后的 raw panic 仍按原规则隔离保留，**不声称已有恢复凭证或可回收**。
- r35 原 image service 增加 Pending initialization：每轮至多推进一个初始化步骤，再推进已有真实 DiT 请求；初始化完成前不入 cohort，取消/关闭等待原 owner 安全终结，未知错误保留原 owner，仅显式 quiescent Rejected 可局部失败。H3 run/serve 共用两阶段 consumer，扩写文本有效且原 causal owner 真正 Retired 后才重新准备 TE/refiner/DiT/cache/全部预算，沿用原 ingress identity 和一个 job；这是源码接线，尚无完整 H3 两阶段成功执行证据。
- r35 native 扩写阶段沿原 BF16 source→F16 compute、分块 prefill/greedy decode/EOS 执行。首次 16 项中 15 通过、1 卡住，线程采样确认卡在 nested block_on，结合 Tokio 协作预算的代码复核修正执行边界；对单次真实 wave 使用 unconstrained，并将 admission maintenance 拆成每次一次 probe、最多一次维护，保留同一 history/原观察事实后返回调度器，下次读取新取消。最终 image generation **249 项全部通过、0 忽略（9.24 秒）**，已包含扩写 19 项、Pending initialization 8 项、auxiliary capacity 7 项及原 Qwen 回归，不能重复累计。首次全组 248 通过、1 失败是测试仍假定 admission 一步完成；只更新 admission 边界驱动，保留实际每个 prefill/decode wave 数与退休断言。
- r35 辅助 causal 阶段的 native 准入沿原驻留 owner、dynamic pool、真实物理可用量和 LRU 释放；保护 pinned/checked-out owner，无 fake decoder target，无独立账本。注册测试初次有 6 处私有观测方法编译错误，改读原 retirement receipts 后通过。实际 Stored owner 的释放/保留与容量算术是本地证据，不能解释为 CUDA 显存或官方模型容量资格。
- r35 CLI 默认扩写 Enabled，要求显式 semantic/tokenizer/BF16 archive 和 causal module 额度；raw 需显式 Disabled。CUDA 产品接同一 tensor+causal runtime、真实 causal family preflight 与三条实际 stream 需求，**CUDA 源码尚未编译/运行**。CLI 配置 **8 项通过（0.02 秒）**；首次误选 binary target 的 0 tests 不计通过，改 lib 后两项因 macOS /var 符号链接的路径断言失败，按原 loader 的 canonical document parent 修正测试后全通过，生产解析行为未改。扩写器 W4/W8 路线仍未实现，不把 BF16 存储/F16 计算声称为 BF16 数值等价。
- r35 默认扩写下的编辑仍缺实际视觉注入和 THW/mRoPE，当前明确拒绝，不静默退回 raw。官方 Transformers v5.20.2 固定源码/合同已在库外校验保存：扩写视觉用 main pooled rows，无 main-TE DeepStack；HF bicubic/antialias 与现 Comfy bilinear，以及位置插值舍入规则不同，后续需独立显式 profile。CPU tiny Qwen 与 Stored image service 的分离测试不能证明整合成功；完整 validated tiny H3 assets fixture 与同 CUDA runtime 的成功 consumer 联测仍待完成，契约保存在 `r35-h3-consumer-test-contract.json`。
- r35 05:28、05:47（上海）两轮无代理原 helper 重试，5090 和 Spark 均在 Mini `100.101.88.54:22` 第一跳约 45 秒超时。末轮日志为 `hardware-r35-final-20261104T214728Z-*`，退出 255、receivedBytes=0；远端命令未执行，停服务仍为 **0**。无完整官方 H3 图片、4MP B2/8MP B1 或 SLO 结果。仅透明压缩本任务 Cargo 产物并核对 SHA/mtime 以应对本机低磁盘余量，未处理用户其他缓存、模型或源码；未暂存、提交、推送。
- r35 稳定源码检查：最终 fmt/diff 通过，workspace all-targets check **通过（1 分 26 秒）**，Clippy `-A warnings` **通过（1 分 35 秒）**，Metal all-targets **编译通过（1 分 23 秒）**。本轮未重跑全量 workspace tests；r33 两次均在执行前链接 ENOSPC，本轮 CLI 构建曾因可用空间跌至约 110MiB 主动中断，压缩本任务已完成产物后恢复并完成针对性测试。全量测试仍未完成，CPU/Metal 编译不能替代 CUDA 编译或运行。
- r35 注册源码快照共 **573 文件、1677271 字节**，SHA-256 `9f33cf338b4f1960f89bd262a7d89279d5856b7d6969353cb4895cdba1867d8b`；逐文件内容 SHA 与 tar 清单核对一致。继续排除本 goal 与两个未注册 CUDA 轨迹草稿，保留两个旧文件删除记录；快照仅存库外、未上传。验证总记录为库外 `r35-verification.json`；目标继续进行，尚不能宣告完整模型支持。


- 2026-11-05 r36：沿原 causal executor 增加通用、不可变的 token 输入契约，区分完整位置表与稀疏视觉行；原 Host 账户按实际分块授信，源 F32 Output 与新 F16 Input 分别持有真实 backing。原 request admission/sequence 保留同一 Arc，重试不得换图；物理 token/KV 位置不与 THW 旋转坐标混用。通用输入 4 项与重试身份 1 项通过，桥接额度精确/少 1B、外账户、错误 class、非有限转换退款由引擎测试覆盖。
- r36 HF 视觉输入采用独立显式 profile：原图先走 Diffusers outer Lanczos，再走 U8 bicubic antialias、F32 归一化与位置表；BF16 权重经已有受管源转为 F32 视觉计算，仅输出扩写器 main pooled rows，不使用主 TE 的 DeepStack。6 项模型测试通过，包含真实 Stored 小型视觉图对独立标量结果及共享权重重绑；不声明 Torch/Torchvision 二进制一致或官方完整权重等价。原已校验 mmap 在视觉/语言源之间共享，不重新打开任意路径；原路径替换后仍持原 mapping 的测试通过。
- r36 按原 tokenizer/template 扩展实际 image-pad 行，构造 THW 与 continuation delta；提示词协议 17 项通过，其中本轮图片/旋转新增 8 项。新 Qwen visual profile 在 embedding 后选择视觉行，仅 full-attention 使用显式 half-split 旋转表，GDN 与物理 KV 位置沿原逻辑；旧纯文本 profile 和 source 规则保留。通用 row-select 统一为零选原行、非零选视觉行，CPU/CUDA 合同一致，产品 mask 仍只产 0/1；CPU 算子 2 项通过。
- r36 真实 Qwen CPU consumer 首次 3 项均输出异常，未更改数学期望。读取实际 Prepared F16 权重的独立标量计算确认单 token 正确、多个 token 错误，进而发现通用内存规划只保护 Fixed 外部输入，而提前上传的 Tokens 位置表仍可与早期内部激活复用。修正为所有外部输入独占其 wave 存储，内部已产出值仍可复用；6 项 liveness 测试通过（含新增 Tokens late-input/alias 及真实 80B/79B 容量边界）。所需 native step 容量随真实新 slots 计算，不保留旧的错误容量数字。
- r36 修后实际 Stored 混合 GDN/full-attention Qwen CPU consumer **4 项全部通过（0.41 秒）**，覆盖真实权重标量结果、视觉替换、整段/分块、prefill/decode 显式位置、双请求交错和取消。运行时完整 lib tests **922 通过、0 失败、5 项既有性能诊断忽略（2.31 秒）**；引擎完整 lib tests **713 通过、0 失败、1 项既有性能诊断忽略（11.85 秒）**，包含图像执行 251 项及上述 4 项，子集不重复累计。CPU 小模型不代表官方 H3、CUDA 或完整生图链路验收。
- r36 视觉组件及 F32→F16 causal bridge 已注册编译，实际 service phase consumer 尚未启用。普通 Result 错误保留原 phase/资源、成功要求原 close ack；原生 panic 仍需接入 TensorComponent 的 retained construction custody，不能用 best-effort Drop 当退休证明。完整阶段还须组合 vision F32 Output 与 causal Input 共存峰，再接共享 run/serve 的原请求身份、取消与初始化调度；当前 Enabled 编辑仍明确拒绝。
- r36 CUDA 源码增加显式 visual profile 与实际 provider/ABI 接线，仍未编译或运行。06:00、06:37 两轮原 helper 去代理探测均在 Mini 第一跳 100.101.88.54:22 超时；末轮日志为 hardware-r36-final-20261104T223723Z-{5090,spark}.log，分别约 45.024/45.030 秒、退出 255、receivedBytes=0。远端命令和停服务仍为 0；无官方 H3 GPU 图片、4MP B2/8MP B1 或 SLO 结果。
- r36 本机两次测试链接受 ENOSPC 影响（engine、CLI），rust-objcopy 产生不完整文件后测试未运行，这两次不计通过。仅处理本任务 Cargo 缓存：保留源码、rlib、metadata、模型与日志，记录并移除需重链的 11 个本任务测试程序，另按 SHA/mtime 校验透明压缩已结束的构建产物；未处理用户其他缓存或模型。编译错误、初次失败与修后日志全部保留在库外 evidence。没有暂存、提交或推送。
- r36 稳定源码检查：fmt/diff 通过，workspace all-targets check **通过（2 分 20 秒）**，Clippy `-A warnings` **通过（2 分 36 秒）**，Metal all-targets **编译通过（1 分 28 秒）**。CLI H3 配置重链后 **8 项通过（0.01 秒）**。全量 workspace tests 本轮未运行；保留先前全量链接 ENOSPC 与本轮两次链接失败证据，当前仅声称已完成的运行时/引擎完整 lib tests 及针对性测试。CUDA 编译/运行未做，Metal 编译不是 GPU 正确性或性能验收。
- r36 注册源码快照共 **619 文件、1886354 字节**，SHA-256 `f8443aa0477522fa3a6ea721c01470c310cf2290c5bee02004047e794c217d0d`；逐文件 SHA 与 tar 清单核对一致。排除本 goal 和两个未注册 CUDA 轨迹草稿，保留两个旧文件删除记录；仅存库外、未上传。完整记录为库外 `r36-verification.json`，目标继续进行。

- 2026-11-05 r37：视觉扩写接入原 Pending initialization，run/serve 共用同一 consumer。实际顺序为原 PNG 严格解码、HF main F32 视觉输出、原组件 Closed、F32→F16 输入桥、原 causal executor 扩写并 Retired、以有效提示词准备原生图/编辑链路。沿用原 Runtime、TensorSerialActor、Host account、pipeline identity 和一个 job；不把扩写视觉组件冒充主 TE 的视觉权重或新建一套执行器。
- r37 组件 retained 初始化把原 phase 与真实 materializer backing 在 native 获取前交入托管；Referenced/关闭失败沿原 root/failure 重试，只有同一原 CloseReceipt 的实际 Closed 才交回 scope。原生获取后的未知 panic 或无法证明退休的失败继续隔离保留，不声称恢复或退款。执行层完整 lib tests **260 通过、0 忽略**，包含新增 7 项构造/关闭/异常/取消所有权测试；原 runtime 关闭失败与隔离 integration **2 项通过**。首次测试 API 编译错误和固定 tensor 测试 provider 未实现实际编码入口均已修正，原日志保留。
- r37 扩写视觉资产冻结原 processor、明确的数值 profile、原 HF family/权重 schema，与语言阶段共享同一已验证 mmap。`ManagedProductionModel::with_prepared` 要求同一冻结 source Arc，并验证新 schema，避免重新打开可变路径或再次哈希整份权重；相关 **6 项模型测试通过**。CLI Enabled 新增显式 `processor_config` 及 typed 视觉输入/patch/family 限额，读取有界且先验证原算法；Disabled 不读取 processor。公共默认视觉 profile 明确为 CUDA F32，另有显式 profile 入口；不把 BF16 存储/F16 causal/F32 vision 声称为官方 BF16 数值等价。
- r37 CUDA 构造按实际已验证资产能力选旧纯文本或新的 visual causal 合同、runtime 配置和 providers，真实 HF vision/visual causal family 并入原同一个 bundle 的 preflight；没有第二个 tensor bundle。CLI H3 配置 **11 项全部通过**。CUDA 代码尚未编译/运行，完整同 Runtime 的视觉→causal→图像链仍待实机验收；Reference F32 vision 与 CpuVNext F16 causal 当前不是同一可执行 CPU 组合，不能拼接两侧结果冒充整链成功。
- r37 Host phase 预算按真实生命周期组合：Input 取“解码 RGBA+视觉预处理”与“causal persistent+wave”较大者；Output 取视觉 F32 与 causal history/readback 较大者。RGBA 在桥接前释放，F32 output 在 causal 安装前释放；权重 backing 保守保留两侧需求之和，staging/native peak 仅在视觉原 close ack 后复用。原 PNG 和已保留输出额度在 phase 外独立计入。真实分配对齐/临界容量/溢出、有效提示词保留编辑参数、同 bytes 异 allocation/异 slice 拒绝共 **4 项通过**。首次容量 fixture 漏算 leading-offset allowance，改用原 `backing_bytes` 查询后通过；生产预算本来就使用该查询。
- r37 新入口的 ingress proof 同时校验 generation 元数据与原 PNG `same_region`；原浅 clone 合法，同内容不同 backing 不可替换。校验发生在消费 phase 或推进状态前，活跃请求身份违例停止 pipeline 并保留原 opaque owner。图片不可变身份已由原 region 证明，不在每个 causal wave 重哈希整张 PNG；最终下游只替换 prompt，原图、seed、尺寸和请求类别保留。
- r37 真实 Stored CPU 视觉 producer/source/bridge **6 项通过**（含 4 项新增、2 项原桥接测试），覆盖同 actor 真实 F32 输出、原 Closed、精确 scope 回收、错误尺寸在 native 前拒绝、profile/限额/源身份及路径替换。完整引擎 lib tests **721 通过、0 失败、1 项既有诊断忽略（11.08 秒）**，上述子集不重复累计；完整 runtime lib tests **922 通过、0 失败、5 项既有诊断忽略（2.25 秒）**。这些是 CPU 与资源生命周期证据，不是官方完整模型 GPU 图片或性能资格。
- r37 稳定源码 fmt/diff 和 workspace all-targets check **通过（2 分 20 秒）**；Clippy `-A warnings` **通过（2 分 37 秒）**，Metal all-targets **编译通过（1 分 28 秒）**。完整 workspace all-targets tests 本轮未重跑，保留此前 ENOSPC 链接失败边界；不能用已完成的几个完整 lib suites 代称全 workspace tests。只透明压缩本任务已结束的 Cargo 产物，核对每个文件 SHA/mtime，未处理用户其他缓存、源码、模型或日志。
- r37 07:08、07:34 两轮原 helper 无代理连接，5090 与 Spark 都卡在 Mini `100.101.88.54:22`；末轮分别 **45.017/45.019 秒、退出 255、receivedBytes=0**。本机 Tailscale Running/Online、Mini 控制面列为 Online，但一次 5 秒 TSMP ping 也无应答；本机 SSH 配置没有 ProxyCommand/ProxyJump。未取得远端 shell，停服务操作仍为 **0**，没有新增 GPU 显存、H3 图片、4MP B2/8MP B1 或 SLO 数据。源码与本 goal 均未暂存、提交或推送。
- r37 注册源码快照共 **638 文件、1950487 字节**，SHA-256 `131754aea19aeba490619981f0aafae7dc3afeb3a149904199b471e188b1098e`；继续排除本 goal 与两个未注册 CUDA 轨迹草稿，记录两个旧文件删除。快照与验证总记录 `r37-verification.json` 仅存库外、未上传；本 goal 保持进行中。
- 2026-11-05 r38：主 TE 构造新增基于已验证 connector 的 typed 入口，真实五层小模型与官方模型共用加载、请求准备和 CUDA preflight 路径；仍校验五个 tap、末层 normalized hidden、实际宽度和原 W4/W8 metadata。旧严格官方构造入口保留。修复 connector 原 `_diffusers_version` 与重建计算配置进行完整 Eq 比较导致官方 metadata 被误拒的问题；现在只比较 tap/hidden/output 计算约定，原版本字段仍进入 metadata SHA。声明 connector 正反合同 **2 项通过**，包含在下述 H3 模型测试中，不重复累计。
- r38 完整小资产复用唯一的原 payload builders，包含 INT8 embedding、混合 W4/W8 TE、F32 connector、BF16 主视觉及三路 DeepStack、24 通道 DiT、原 CNN encoder 与 INT8 VAE decoder。三份真实 mmap source 通过正式 `PH3ImageAssets::prepare`，并验证生成/编辑 recipes、混合请求 DiT 各阶段的原共享 schema、VAE 4D 与 DiT 5D latent 的真实尺寸和字节连接关系。完整资产 **2 项通过**；这些是加载和计算图准备证据，没有创建 CUDA Runtime，也不是完整出图证据。
- r38 原模型回归：H3/VAE **191 通过、0 失败、0 忽略（9.28 秒）**；Comfy/HF 视觉 **25 通过、0 失败、0 忽略（0.70 秒）**；完整引擎 lib **723 通过、0 失败、1 项既有诊断忽略（11.51 秒）**，已包含上述完整资产 2 项。首轮 models 编译因旧 encoder 执行测试缺少抽取后 `bytes` helper 的可见性/引用而失败，保留原唯一函数并修正可见性、重新导出后通过；旧数值、shape、dtype 和 rounding 与 r37 快照核对未变。
- r38 注册三个真实 CUDA ignored 入口：两项 `cuda_dit_refine_euler` 轨迹测试沿原 actor、snapshot、packed step 和 Euler，对照 B1 并检查加入/完成/取消/替换及 late successor 拒绝；修复 stream 数量与重复计入 Host 对齐的 fixture 问题。一项 `cuda_complete_h3_expansion_disabled_generate_edit_and_service_match` 通过正式工厂，在同一 Runtime 下先生成/编辑，再移交原 service 对比同 seed 的实际 F32/像素输出及关闭/Host 退款。三项均**尚未进行 CUDA 编译和实跑**；完整入口明确扩写 Disabled，部分 tiny TE projection 为零，不建立官方数值、提示词语义、Enabled 整链、全局 VRAM 归零或并发性能资格。
- r38 稳定源码 fmt/diff、workspace all-targets check **通过（1 分 08 秒）**，Clippy `-A warnings` **通过（1 分 14 秒）**，Metal all-targets **编译通过（1 分 11 秒）**。首轮 fmt 仅发现新 tests 模块排序，调整后通过。全 workspace all-targets tests 本轮未重跑：此前 ENOSPC 链接失败仍是未完成项，本机可用空间仍不足 1GiB；没有用上述局部测试替代全量验收。只透明压缩本任务四个已结束 Cargo 产物并核对 SHA/mtime，保留全部源代码、模型、日志和其他缓存。
- r38 07:44 四个已知无代理 TCP 入口（Mini Tailscale/LAN、5090 LAN、Spark LAN）均超时。07:57 原 helper 两路无代理 SSH 再次卡在 Mini `100.101.88.54:22`，分别 **45.024/45.015 秒，退出 255，receivedBytes=0**；未进入远端 shell，停服务仍为 **0**。已请求可用的新入口或 Mini 恢复信息；已有停占用服务授权保留，未重复请求操作批准。官方 H3 GPU 图片、4MP B2/8MP B1、Qwen 实机回归和 SLO 仍无新增证据。
- r38 连续批处理观测审查确认现 bench 的 HTTP 并发响应不能独立证明真实 B≥2；原 receipt 仅 last batch/步号也不足以重建全部成员变化。已在库外记录基于原 cohort job、实际提交 times、forward fence 和原子 successor commit 的有界事件设计，尚未实现；继续保留 `g3_status=not_evaluated`。后续需补齐实际服务轨迹和 Enabled 同 Runtime 测试，再进行实机容量/性能验收。
- r38 注册源码快照共 **650 文件、1970752 字节**，SHA-256 `cb770f0d6fd9376f9b2955baf76f79efeb4c27409037caa8396050502455ff8b`；三个 CUDA 入口已纳入，仍排除本 goal，记录原两个删除文件。逐文件 SHA、tar 清单和 1MiB 源文件边界检查通过；快照与 `r38-verification.json` 仅存库外、未上传。未暂存、提交或推送；本 goal 仍进行中。

- 2026-11-05 r39：扩写 Enabled 的完整 CUDA 入口已注册，与 Disabled 入口共用正式工厂、同一 Runtime、原始资产与实际 service 配置；生成和编辑经过原视觉→causal→图像消费链。两项完整入口与两项 DiT 轨迹入口均加入实际追踪断言，共 **4 项 CUDA 测试仍未编译和实跑**。tiny TE 中零值 projection 的既有边界保留，不建立官方提示词语义或数值等价资格。
- r39 有界 collector 只持有原 Host 的一份真实 Output backing，事件借用原 request、cohort、plan/program、实际 time basis 和 row runs；仅在真实发布、forward fence/solver commit、步骤边界退出后记录，不由客户端并发数或最后一个 batch receipt 推断合批。原 shutdown 成功才 seal；seal 本身不证明全部外部资源已退休。满缓冲、序列化失败及 panic 保留完整前缀并标记 truncated，不替换原推理结果。collector **8 项边界测试通过**，包含在下述 engine suite 中。
- r39 Rust analyzer 对 schema/session/连续序号、原发布与 commit 成员、step/sigma 连续性、packed 行范围、RLE 与时间索引、实际退出及 footer 数量/字节做有界校验。只有实际 commit 计入联合执行、混进度和补位事实；固定参考/音频时间不会冒充异步进度。缺 header 的组件轨迹、未 seal、截断或丢记录均不建立完整证据；metadata SHA 不声称认证完整权重。解析器 **8 项通过（0.81 秒）**；首次逐字节截断测试暴露 serde 忽略字段的数字扫描错误分类，修为仅由有界普通数字解析明确确认 EOF 才认定不完整，非法数字仍拒绝。当前提供 Rust API，未改现有 HTTP benchmark 的 G3/SLO 资格。
- r39 run/serve 的 H3 配置新增可选 `continuous_trace: {"path":"steps.jsonl","maximum_bytes":1048576}`，默认不分配。原 Host Output 容量必须容纳其实际对齐 backing；文件在 native 构造前 create_new，原关闭结果确定后导出，失败前缀如实保留未 seal 状态，诊断写入错误不吞掉原执行/退休错误。profile 拒绝无消费者的追踪配置。CLI 首轮发现空路径先被解析成配置目录而漏拒，已改为相对路径展开前校验；修后配置与 sink **3 项通过**，包含在下述 CLI 图像测试中。
- r39 修复确定的永久等待边界：追踪 backing 在整个 producer 生命周期内不会因排空普通响应而释放。7 处 H3 阶段容量证明计入原物理占用，service/scheduler 和 HTTP ResponseGate 经通用 `retained_output_bytes` 查询保留同一固定 baseline；实际 reserve/exchange 仍只申请新工作，未新建账户、类别或重复授信。真实 Host 边界验证不足时有限拒绝、精确容量可执行、普通 caller Output 最后所有者释放可重试，以及 HTTP 在排队/推理前返回容量错误。engine 定向 **12 项**、HTTP 定向 **4 项**通过，均属于下述完整受影响 suites，不重复累计。
- r39 完整 engine lib **737 通过、0 失败、1 项既有诊断忽略（12.05 秒）**，包含新增原 fence 后完成/取消/失败观测与 Enabled 原资产元数据准备测试；HTTP 图像模块 **31 通过（1.03 秒）**，CLI 图像功能 **66 通过（0.75 秒）**。模型实际借用原 generation/edit 行路由及各阶段指针/时间配置测试 **1 项通过（1.03 秒）**。首轮 collector 编译遗漏新 execution 字段和 executable plan 访问层级、service 测试 fixture 缺 required compatible 方法均已修正，失败与修后日志均保留；未将 CPU/协议测试冒充 CUDA 正确性。
- r39 稳定源码 fmt/diff 检查通过，workspace all-targets check **通过（1 分 31 秒）**，Clippy `-A warnings` **通过（1 分 39 秒）**，Metal all-targets **编译通过（1 分 12 秒）**。本轮未重跑全 workspace all-targets tests，先前 ENOSPC 链接失败仍属于未完成项，本机可用空间约 0.5GiB。仅对本任务已结束 Cargo 产物做逐文件 SHA/mtime 验证的透明压缩，所有记录保存在库外 evidence。
- r39 08:39 原 helper 清除代理后连接 5090，仍在 Mini `100.101.88.54:22` 超时，**45.022 秒、退出 255、receivedBytes=0**；08:59:40 对 Mini Tailscale/LAN、5090 LAN、Spark LAN 的原始 TCP 再探测全部约 5 秒超时。没有远端 shell，停占用服务操作仍为 **0**。已保留用户对 5090 停服务和两台设备使用的授权；没有新增官方图片、实机容量、Qwen GPU 回归或性能数据。
- r39 注册源码快照 **665 文件、2013323 字节**，SHA-256 `f935f6f7cb2cd121f4b8e70bc5ea437e311254c6be00c71fc8ee2e41271d12de`；排除本 goal，记录原两个删除文件，逐文件 SHA、tar 清单与 1MiB 源文件边界核对通过。快照和 `r39-verification.json` 仅存库外、未上传。未暂存、提交或推送；目标保持进行中，下一阶段需要 CUDA 环境与正式模型实机验收。
