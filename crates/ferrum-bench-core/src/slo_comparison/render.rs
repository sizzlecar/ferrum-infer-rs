use super::evidence::memory_peak_issues;
use super::*;
use crate::slo::SloStatus;
use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkdownLanguage {
    English,
    Chinese,
}

fn markdown_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace(['\n', '\r'], " ")
}
fn number(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "Unknown".into())
}

fn absolute_slo_status(sources: &[&ObservedArmSource], expected: usize) -> SloStatus {
    if sources
        .iter()
        .any(|source| source.absolute_slo_status == SloStatus::Fail)
    {
        SloStatus::Fail
    } else if sources.len() != expected
        || sources
            .iter()
            .any(|source| source.absolute_slo_status != SloStatus::Pass)
    {
        SloStatus::Unknown
    } else {
        SloStatus::Pass
    }
}

fn memory_cell(
    cell: &CellComparison,
    candidate: bool,
    kind: MemoryMeasurement,
    contract: &FrozenComparisonContract,
) -> String {
    let mut raw_peak = None::<u64>;
    let mut valid_repeats = 0;
    for pair in &cell.pairs {
        let (memory, source) = if candidate {
            (
                pair.candidate_memory.as_ref(),
                pair.candidate_source.as_ref(),
            )
        } else {
            (pair.baseline_memory.as_ref(), pair.baseline_source.as_ref())
        };
        let Some(memory) = memory else {
            continue;
        };
        let peak = match kind {
            MemoryMeasurement::SampledDeviceAllocation => &memory.device_allocation,
            MemoryMeasurement::SampledOsPhysicalFootprint
            | MemoryMeasurement::ProcessPeakPhysicalFootprint => &memory.os_footprint,
            MemoryMeasurement::ProcessMaximumRss => &memory.maximum_rss,
        };
        if let Some(bytes) = peak.peak_bytes {
            raw_peak = Some(raw_peak.unwrap_or(0).max(bytes));
        }
        if source.is_some_and(|source| {
            memory_peak_issues(peak, kind, &source.execution, &contract.memory).is_empty()
        }) {
            valid_repeats += 1;
        }
    }
    let expected = contract.pairs.len();
    let gib = raw_peak.map(|bytes| bytes as f64 / 1_073_741_824.0);
    if valid_repeats == expected && cell.pairs.len() == expected {
        number(gib)
    } else if let Some(gib) = gib {
        format!("Unknown (unverified max {gib:.3}; coverage {valid_repeats}/{expected})")
    } else {
        "Unknown".into()
    }
}

impl SloComparisonReport {
    /// Both README languages and the website can render the same report. ITL is
    /// pooled visible text-event spacing; memory columns are separate high-water
    /// observations and must never be added as if they were disjoint allocations.
    pub fn to_markdown(&self, language: MarkdownLanguage) -> String {
        let zh = language == MarkdownLanguage::Chinese;
        let mut output = String::new();
        let title = if zh {
            "冻结合同"
        } else {
            "Frozen contract"
        };
        let _ = writeln!(
            output,
            "{title}: `{}` · {:?}\n",
            self.frozen_contract_sha256, self.status
        );
        let footprint_method = match &self.contract.memory.os_footprint {
            OsFootprintPolicy::Sampled(policy) => format!(
                "{}: {} ({} ms; max gap {} ns)",
                if zh { "OS footprint 采样窗口" } else { "OS footprint sampling window" },
                markdown_text(&policy.window), policy.interval_ms, policy.max_sample_gap_ns
            ),
            OsFootprintPolicy::ProcessLifetime(_) => if zh {
                "OS footprint 与最大 RSS：macOS time -l 整个进程寿命峰值，含加载、预热、测量和关闭；不是采样或仅测量窗口峰值。"
            } else {
                "OS footprint and maximum RSS: macOS time -l whole process lifetime peaks, including load, warmup, measurement and shutdown; not sampled or measurement-only peaks."
            }.into(),
        };
        let _ = writeln!(output, "{footprint_method}\n");
        let _ = writeln!(
            output,
            "{}\n",
            if zh {
                "TTFT / TPOT / 可见 ITL 的 P50/P99 为各轮分位数的算术均值；可见 ITL 指非空 SSE 文本事件间隔。TPOT 结束于最后可见文本。比值表使用逐配对比值均值，不能由主表两行均值相除代替。绝对 SLO 分别按每个实现的原始请求重算；比较状态另列。内存分别取已声明各轮最大值；设备分配为采样高水位，OS footprint 按上方明确的方法，RSS 为进程最大值，不相加。正常内存数字要求所有轮次的证据完整性与测量边界符合合同；采样指标另需符合采样覆盖要求。Unknown 后的 unverified max 仅为未核实的原始观测，coverage 为证据完整的轮数。Unknown 不代表零。"
            } else {
                "TTFT / TPOT / visible ITL P50/P99 are arithmetic means of per-repeat percentiles. Visible ITL measures nonempty SSE text-event gaps; TPOT ends at last visible text. The ratio table averages paired ratios, not ratios of these means. Absolute SLO is recomputed separately for each arm; comparison status is separate. Memory columns independently take maxima across declared repeats: sampled device allocation, OS footprint using the explicit method above, and process maximum RSS; never sum them. Plain memory values require complete evidence and contract-compliant measurement boundaries in every repeat; sampled metrics additionally require sampling coverage. An unverified max after Unknown is only a raw observation; coverage counts complete repeats. Unknown is not zero."
            }
        );
        let _ = writeln!(output, "| {} | C | {} | {} | TTFT P50 / P99 ms | TPOT P50 / P99 ms/token | {} | {} | {} | {} | RSS GiB | {} | {} | {} |",
            if zh { "范围" } else { "Scope" }, if zh { "实现" } else { "Implementation" }, if zh { "完整配对轮数" } else { "Complete pairs" },
            if zh { "可见 ITL P50 / P99 ms" } else { "Visible ITL P50 / P99 ms" }, if zh { "成功 usage token/s" } else { "Successful usage token/s" },
            if zh { "设备分配 GiB" } else { "Device allocation GiB" }, if zh { "OS footprint GiB" } else { "OS footprint GiB" },
            if zh { "请求 / gap / 失败 / 拒绝 / pending" } else { "Requests / gaps / failed / rejected / pending" },
            if zh { "本实现绝对 SLO" } else { "Arm absolute SLO" }, if zh { "比较状态" } else { "Comparison status" });
        output.push_str("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n");
        for cell in &self.cells {
            let complete = cell
                .pairs
                .iter()
                .filter(|pair| pair.evidence_status == ComparisonStatus::ObservedPass)
                .count();
            for candidate in [false, true] {
                let arm = if candidate {
                    &self.contract.candidate
                } else {
                    &self.contract.baseline
                };
                let values: BTreeMap<_, _> = ComparisonMetric::ALL
                    .into_iter()
                    .map(|metric| {
                        let values: Vec<_> = cell
                            .pairs
                            .iter()
                            .filter_map(|pair| {
                                pair.ratios.get(&metric).map(|v| {
                                    if candidate {
                                        v.candidate
                                    } else {
                                        v.baseline
                                    }
                                })
                            })
                            .collect();
                        (
                            metric,
                            (values.len() == self.contract.pairs.len()).then(|| mean(&values)),
                        )
                    })
                    .collect();
                let sources: Vec<_> = cell
                    .pairs
                    .iter()
                    .filter_map(|pair| {
                        if candidate {
                            pair.candidate_source.as_ref()
                        } else {
                            pair.baseline_source.as_ref()
                        }
                    })
                    .collect();
                let counts = if sources.len() == self.contract.pairs.len() {
                    format!(
                        "{} / {} / {} / {} / {}",
                        sources.iter().map(|s| s.offered_requests).sum::<usize>(),
                        sources
                            .iter()
                            .map(|s| s.observed_visible_gaps)
                            .sum::<usize>(),
                        sources.iter().map(|s| s.failed_requests).sum::<u64>(),
                        sources.iter().map(|s| s.rejected_requests).sum::<u64>(),
                        sources.iter().map(|s| s.pending_requests).sum::<u64>()
                    )
                } else {
                    "Unknown".into()
                };
                let _ = writeln!(output, "| {:?} | {} | {} | {}/{} | {} / {} | {} / {} | {} / {} | {} | {} | {} | {} | {} | {:?} | {:?} |", cell.scope, cell.concurrency,
                    markdown_text(&arm.implementation), complete, self.contract.pairs.len(), number(values[&ComparisonMetric::TtftP50]), number(values[&ComparisonMetric::TtftP99]),
                    number(values[&ComparisonMetric::TpotP50]), number(values[&ComparisonMetric::TpotP99]), number(values[&ComparisonMetric::VisibleItlP50]), number(values[&ComparisonMetric::VisibleItlP99]),
                    number(values[&ComparisonMetric::SuccessfulUsageOutputTps]), memory_cell(cell, candidate, MemoryMeasurement::SampledDeviceAllocation, &self.contract),
                    memory_cell(cell, candidate, self.contract.memory.os_footprint.measurement(), &self.contract),
                    memory_cell(cell, candidate, MemoryMeasurement::ProcessMaximumRss, &self.contract), counts, absolute_slo_status(&sources, self.contract.pairs.len()), cell.status);
            }
        }
        let _ = writeln!(
            output,
            "\n| C | {} | {} | {} | {} | {} |\n|---:|---|---:|---|---:|---|",
            if zh { "指标" } else { "Metric" },
            if zh {
                "逐对比值均值"
            } else {
                "Mean paired ratio"
            },
            if zh {
                "观察范围（非置信区间）"
            } else {
                "Observed range (not CI)"
            },
            if zh { "合同边界" } else { "Contract limit" },
            if zh {
                "描述性状态"
            } else {
                "Descriptive status"
            }
        );
        for cell in &self.cells {
            for (metric, result) in &cell.metrics {
                let range = result
                    .observed_ratio_range
                    .map(|(lo, hi)| format!("{lo:.3}–{hi:.3}"))
                    .unwrap_or_else(|| "Unknown".into());
                let _ = writeln!(
                    output,
                    "| {} | {} | {} | {} | {} {:.3} | {:?} |",
                    cell.concurrency,
                    metric.label(),
                    number(result.mean_paired_ratio),
                    range,
                    if metric.is_throughput() { "≥" } else { "≤" },
                    result.limit,
                    result.status
                );
            }
        }
        if let Some(inference) = &self.computed_bootstrap {
            render_bootstrap(&mut output, inference, zh);
        }
        if let Some(eligibility) = &self.inference_eligibility {
            let _ = writeln!(
                output,
                "\n{}: {}; plan=`{}`; pilot=`{}`; {}\n",
                if zh {
                    "原始 pilot 规划选定配对轮数"
                } else {
                    "Raw-pilot planned paired repetitions"
                },
                eligibility.selected_pairs,
                eligibility.plan_sha256,
                eligibility.pilot_source_sha256,
                if zh {
                    "资格验证覆盖规划、支持量与采集诊断，不证明真实 IID 或无条件覆盖率。"
                } else {
                    "Eligibility verifies planning, support and acquisition diagnostics, not real-world IID or unconditional coverage."
                }
            );
        }
        let _ = writeln!(
            output,
            "\n{}\n",
            if self.inference_eligibility.is_some() && zh {
                "ProofPass 仅表示在声明的独立、可交换整轮配对假设与近似同时置信区间下通过冻结合同，不是硬保证。ObservedPass 仍仅为描述性结果。输出检查不证明语义质量。"
            } else if self.inference_eligibility.is_some() {
                "ProofPass means the frozen contract passed under declared independent, exchangeable paired-block assumptions and approximate simultaneous bounds; it is not a hard guarantee. ObservedPass remains descriptive. Output checks do not establish semantic quality."
            } else if zh {
                "统计推断尚未验证，ObservedPass 仅为描述性点估计达标，不是性能优势证明。输出检查仅覆盖协议/空输出，不证明语义质量。"
            } else {
                "Statistical inference is unverified. ObservedPass describes point estimates; it is not proof of superiority. Output checks cover protocol/empty output, not semantic quality."
            }
        );
        for issue in &self.issues {
            let _ = writeln!(output, "- {}", markdown_text(issue));
        }
        for cell in &self.cells {
            for issue in &cell.issues {
                let _ = writeln!(output, "- C{}: {}", cell.concurrency, markdown_text(issue));
            }
            for pair in &cell.pairs {
                for issue in &pair.issues {
                    let _ = writeln!(
                        output,
                        "- C{} / {}: {}",
                        cell.concurrency,
                        markdown_text(&pair.pair_id),
                        markdown_text(issue)
                    );
                }
            }
        }
        output
    }
}

fn render_bootstrap(output: &mut String, inference: &ComputedBootstrapInference, zh: bool) {
    let eligible =
        inference.calibration == BootstrapCalibrationStatus::EligibleUnderDeclaredAssumptions;
    let _ = writeln!(
        output,
        "\n{}: `{}`; {}: {}; α={:.6}; {}={:.6}; seed={}; B={}; calibration={:?}; status={:?}.\n",
        if zh {
            "整轮配对 bootstrap（近似区间）"
        } else {
            "Paired-cluster bootstrap (approximate bounds)"
        },
        markdown_text(&inference.method_id),
        if zh {
            "完整主比较族大小"
        } else {
            "Complete primary family size"
        },
        inference.family_size,
        inference.family_alpha,
        if zh {
            "模拟误差预算"
        } else {
            "Monte Carlo error budget"
        },
        inference.monte_carlo_error_budget,
        inference.seed,
        inference.resamples,
        inference.calibration,
        inference.status,
    );
    let _ = writeln!(
        output,
        "| C | {} | {} | {} | {} | {} |\n|---:|---|---:|---|---:|---|",
        if zh { "指标" } else { "Metric" },
        if zh {
            "逐对比值均值"
        } else {
            "Mean paired ratio"
        },
        if eligible && zh {
            "单侧近似界（声明假设下）"
        } else if eligible {
            "Approximate one-sided bound (under declared assumptions)"
        } else if zh {
            "单侧界（未校准）"
        } else {
            "One-sided bound (uncalibrated)"
        },
        if zh { "相对宽度" } else { "Relative width" },
        if zh {
            "声明精度满足 / 严格越过阈值（均非证明）"
        } else {
            "Declared precision met / strict limit cleared (neither is proof)"
        }
    );
    for cell in &inference.cells {
        for (metric, bound) in &cell.metrics {
            let _ = writeln!(
                output,
                "| {} | {} | {:.3} | {:?} {} | {} | {} / {} |",
                cell.concurrency,
                metric.label(),
                bound.point_estimate,
                bound.direction,
                number(bound.one_sided_bound),
                number(bound.relative_bound_width),
                bound.precision_target_met,
                bound.strict_limit_cleared
            );
        }
    }
    let _ = writeln!(
        output,
        "\n{}",
        if eligible && zh {
            "整轮配对是抽样单位，轮内请求与可见间隔不视为 IID。原始 pilot 资格已验证，独立性/可交换性和 bootstrap 近似覆盖仍是显式假设；适用范围仅为冻结工作负载的逐对比值算术均值。"
        } else if eligible {
            "Complete paired runs are resampling units, not IID requests or gaps. Raw-pilot eligibility is verified; independence, exchangeability and approximate bootstrap coverage remain explicit assumptions. Scope is the arithmetic mean of paired ratios for the fixed workload."
        } else if zh {
            "区间以独立整轮配对为抽样单位，轮内请求与可见间隔不视为 IID；只适用于冻结工作负载的逐对比值算术均值。P99 支持数和 pilot 引用仍为声明，校准未验证；不产生 ProofPass。"
        } else {
            "Resampling units are independent complete paired runs, not IID requests or visible gaps. Bounds concern arithmetic mean paired ratios for the frozen workload. P99 support counts and pilot references remain declarations; calibration is unverified and cannot produce ProofPass."
        }
    );
    for issue in &inference.issues {
        let _ = writeln!(output, "- {}", markdown_text(issue));
    }
}
