//! Frozen actual F16 attention choices. Worker projection has no PSO access.
use super::super::super::linear::FrozenLinear;
use super::super::super::primitives::FrozenPrimitive;
use super::*;
use ferrum_interfaces::vnext::{DeviceObservationTemplate, FrozenObservationInput};

struct Io {
    rms: FrozenPrimitive,
    input: FrozenLinear,
    output: FrozenLinear,
    residual: FrozenPrimitive,
}
impl Io {
    fn freeze(
        l: &MetalLinearPipelines,
        p: &MetalPrimitivePipelines,
        hidden: ElementType,
        v: Projection,
    ) -> Option<Self> {
        let rms = FrozenPrimitive::rms(p, hidden, ElementType::F16, v.tokens, v.hidden, v.epsilon)?;
        let residual = FrozenPrimitive::residual(p, hidden, v.tokens.checked_mul(v.hidden)?)?;
        let mut input = FrozenLinear::empty();
        for &launch in &v.launches[..3] {
            input.projection(l, launch, None)?;
        }
        let mut output = FrozenLinear::empty();
        output.projection(l, v.launches[3], None)?;
        Some(Self {
            rms,
            input,
            output,
            residual,
        })
    }
    fn input(&self, b: &mut SelectedCommandCostBuilderV1, scratch: u64) -> Option<()> {
        self.rms.append(b, scratch)?;
        self.input.append(b, scratch)
    }
    fn output(&self, b: &mut SelectedCommandCostBuilderV1, scratch: u64) -> Option<()> {
        self.output.append(b, scratch)?;
        self.residual.append(b, scratch)
    }
    fn variable_bytes(&self) -> Option<usize> {
        self.input
            .variable_bytes()?
            .checked_add(self.output.variable_bytes()?)
    }
    fn occurrences(&self) -> Option<u64> {
        self.input
            .occurrences()?
            .checked_add(self.output.occurrences()?)?
            .checked_add(2)
    }
}
#[derive(Clone, Copy)]
struct Attention {
    plan: AttentionDispatchPlan,
    entry: (&'static str, u32),
    reduce: Option<(&'static str, u32)>,
}
impl Attention {
    fn freeze(a: &MetalCausalAttentionPipelines, p: &CausalAttentionParams) -> Option<Self> {
        let plan = a.dispatch_plan(p);
        Some(Self {
            plan,
            entry: selected_entry(a, p, plan.kind, false)?,
            reduce: if plan.kind == AttentionDispatchKind::GroupedDecode {
                Some(selected_entry(a, p, plan.kind, true)?)
            } else {
                None
            },
        })
    }
    fn append(
        &self,
        b: &mut SelectedCommandCostBuilderV1,
        p: &CausalAttentionParams,
        scratch: u64,
    ) -> Option<()> {
        kernel(
            b,
            self.entry.0,
            self.entry.1,
            p,
            pairs(p)?,
            rectangular(p, self.plan.kind)?,
            u64::from(p.head_dim),
            self.plan.threadgroups,
            self.plan.threads_per_threadgroup,
            self.plan.threadgroup_memory_bytes,
            scratch,
        )?;
        if let Some((entry, special)) = self.reduce {
            let n = u64::from(p.query_heads).checked_mul(u64::from(p.head_dim))?;
            kernel(
                b,
                entry,
                special,
                p,
                n,
                n,
                grouped_decode_partitions(p),
                [u64::from(p.query_heads), 1, 1],
                [SIMD_THREADS, 1, 1],
                [grouped_decode_reduce_threadgroup_memory_bytes(), 0],
                scratch,
            )?;
        }
        Some(())
    }
}
struct FrozenRow {
    params: CausalAttentionParams,
    attention: Attention,
    io: Option<Io>,
}
struct Template {
    packed: Option<Io>,
    rows: Vec<FrozenRow>,
    batched: bool,
    independent: bool,
    tokens: u64,
    scratch: u64,
    capture: ferrum_types::SloStructuredCostCapture,
    occurrences: u64,
}
pub(in crate::backend::metal::vnext_ops) fn payload_upper(
    dispatches: u64,
    rows: usize,
) -> Option<usize> {
    std::mem::size_of::<Template>()
        .checked_add(2 * std::mem::size_of::<usize>())?
        .checked_add(
            crate::backend::metal::vnext_runtime::observation_capacity_upper(rows)?.checked_mul(
                std::mem::size_of::<FrozenRow>().checked_add(std::mem::size_of::<Row>())?,
            )?,
        )?
        .checked_add(FrozenLinear::variable_upper(
            dispatches,
            rows.checked_add(1)?.checked_mul(2)?,
        )?)
}
impl DeviceObservationTemplate for Template {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        let mut n = std::mem::size_of::<Self>()
            .checked_add(2 * std::mem::size_of::<usize>())?
            .checked_add(
                self.rows
                    .capacity()
                    .checked_mul(std::mem::size_of::<FrozenRow>())?,
            )?;
        if let Some(io) = &self.packed {
            n = n.checked_add(io.variable_bytes()?)?;
        }
        for row in &self.rows {
            if let Some(io) = &row.io {
                n = n.checked_add(io.variable_bytes()?)?;
            }
        }
        Some(n)
    }
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(self.occurrences)?
            .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
    }
    fn project(
        &self,
        input: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        if input.tokens() != self.tokens
            || !input.participant_ranges().is_empty()
            || !input.source_ranges().is_empty()
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let mut b =
            crate::backend::metal::vnext_runtime::selected_cost_builder(self.capture, self.tokens);
        let projected = (|| {
            if let Some(io) = &self.packed {
                io.input(&mut b, self.scratch)?;
                if self.independent {
                    b.independent_attention_rows_v2(self.rows.iter(), |b, row| {
                        prepare(b, &row.params, self.scratch)
                            .and_then(|()| row.attention.append(b, &row.params, self.scratch))
                            .ok_or(StatisticalEvidenceUnknown::MissingProducer)
                    })
                    .ok()?;
                } else {
                    for row in &self.rows {
                        prepare(&mut b, &row.params, self.scratch)?;
                        if !self.batched {
                            row.attention.append(&mut b, &row.params, self.scratch)?;
                        }
                    }
                }
                if self.batched {
                    for chunk in self.rows.chunks(GROUPED_BATCH_ROWS) {
                        batched_frozen(&mut b, chunk, self.scratch)?;
                    }
                }
                io.output(&mut b, self.scratch)?;
            } else {
                for row in &self.rows {
                    let io = row.io.as_ref()?;
                    io.input(&mut b, self.scratch)?;
                    prepare(&mut b, &row.params, self.scratch)?;
                    row.attention.append(&mut b, &row.params, self.scratch)?;
                    io.output(&mut b, self.scratch)?;
                }
            }
            Some(())
        })();
        projected.ok_or(StatisticalEvidenceUnknown::Overflow)?;
        Ok(vec![Some(b.finish()?)])
    }
}
fn batched_frozen(
    b: &mut SelectedCommandCostBuilderV1,
    rows: &[FrozenRow],
    scratch: u64,
) -> Option<()> {
    let p = &rows.first()?.params;
    let n = u64::try_from(rows.len()).ok()?;
    let mut logical = 0u64;
    let mut padded = 0u64;
    let mut partitions = 0u64;
    let mut reduce_work = 0u64;
    for row in rows {
        let v = &row.params;
        logical = logical.checked_add(pairs(v)?)?;
        padded = padded.checked_add(rectangular(v, AttentionDispatchKind::GroupedDecode)?)?;
        partitions = partitions.max(grouped_decode_partitions(v));
        reduce_work = reduce_work.checked_add(
            u64::from(v.query_heads)
                .checked_mul(u64::from(v.head_dim))?
                .checked_mul(grouped_decode_partitions(v))?,
        )?;
    }
    let plan = grouped_decode_attention_dispatch_plan(p);
    kernel(
        b,
        "vnext_causal_attention_decode_batched_partial_f16",
        p.head_dim,
        p,
        logical,
        padded,
        u64::from(p.head_dim),
        [partitions, u64::from(p.key_value_heads), n],
        [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
        plan.threadgroup_memory_bytes,
        scratch,
    )?;
    kernel(
        b,
        "vnext_causal_attention_decode_batched_reduce_f16",
        p.head_dim,
        p,
        reduce_work,
        reduce_work,
        1,
        [u64::from(p.query_heads), 1, n],
        [SIMD_THREADS, 1, 1],
        [grouped_decode_reduce_threadgroup_memory_bytes(), 0],
        scratch,
    )
}
#[allow(clippy::too_many_arguments)]
pub(in crate::backend::metal::vnext_ops) fn freeze(
    a: &MetalCausalAttentionPipelines,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    packed: Option<Projection>,
    batched_grouped: bool,
    independent_rows: bool,
    rows: &[Row],
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    if a.kv_type != ElementType::F16
        || rows.is_empty()
        || rows.len() > ferrum_interfaces::execution_cost::MAX_COST_ROWS
        || rows
            .iter()
            .try_fold(0u64, |n, r| n.checked_add(u64::from(r.params.tokens)))?
            != tokens
    {
        return None;
    }
    if batched_grouped && (packed.is_none() || rows.len() < 2) {
        return None;
    }
    let independent = packed.is_some()
        && independent_rows
        && !batched_grouped
        && cost_route::Capabilities::from(a)
            .may_group_independent_decode_rows(rows.iter().map(|r| &r.params));
    let packed = match packed {
        Some(v) => {
            if u64::from(v.tokens) != tokens {
                return None;
            }
            Some(Io::freeze(l, p, hidden, v)?)
        }
        None => None,
    };
    let mut frozen = Vec::new();
    frozen.try_reserve_exact(rows.len()).ok()?;
    let mut occurrences = packed.as_ref().map_or(Some(0), Io::occurrences)?;
    for row in rows {
        let attention = Attention::freeze(a, &row.params)?;
        let io = if packed.is_none() {
            Some(Io::freeze(l, p, hidden, row.projection)?)
        } else {
            None
        };
        occurrences = occurrences
            .checked_add(io.as_ref().map_or(Some(0), Io::occurrences)?)?
            .checked_add(1)?;
        if !batched_grouped {
            occurrences = occurrences.checked_add(1 + u64::from(attention.reduce.is_some()))?;
        }
        frozen.push(FrozenRow {
            params: row.params,
            attention,
            io,
        });
    }
    if batched_grouped {
        for chunk in frozen.chunks(GROUPED_BATCH_ROWS) {
            let first = &chunk[0].params;
            a.specialization(first)?.batched_grouped.as_ref()?;
            if chunk.iter().any(|r| {
                r.attention.plan.kind != AttentionDispatchKind::GroupedDecode
                    || r.params.head_dim != first.head_dim
                    || r.params.query_heads != first.query_heads
                    || r.params.key_value_heads != first.key_value_heads
            }) {
                return None;
            }
            occurrences = occurrences.checked_add(2)?;
        }
    }
    Some(Arc::new(Template {
        packed,
        rows: frozen,
        batched: batched_grouped,
        independent,
        tokens,
        scratch,
        capture: l.structured_capture(),
        occurrences,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::super::super::linear::selected_q4_observation_fixture;
    use super::*;
    fn params(position_start: u32) -> CausalAttentionParams {
        CausalAttentionParams {
            page_elements: 32768,
            page_count: 8,
            position_start,
            tokens: 1,
            query_heads: 16,
            key_value_heads: 4,
            head_dim: 256,
            rope_dim: 64,
            query_projection_stride: 8192,
            query_head_stride: 512,
            kv_projection_stride: 1024,
            output_gate: 1,
            rope_interleaved: 1,
            attention_simdgroups: 16,
            epsilon: 1e-6,
            rope_theta: 10_000_000.0,
        }
    }
    fn projection(tokens: u32) -> Projection {
        let make =
            |input, output| selected_q4_observation_fixture(u64::from(tokens), input, output);
        Projection {
            tokens,
            hidden: 256,
            epsilon: 1e-6,
            launches: [
                make(256, 8192),
                make(256, 1024),
                make(256, 1024),
                make(4096, 256),
            ],
        }
    }
    #[test]
    fn causal_observation_frozen_actual_choices_match_ordered_independent_and_batched_routes() {
        let device = Device::system_default().expect("real Metal selector catalog");
        let a = MetalCausalAttentionPipelines::new(&device).unwrap();
        let l = MetalLinearPipelines::new(&device)
            .unwrap()
            .with_structured_capture(ferrum_types::SloStructuredCostCapture::HostSettledV1);
        let p = MetalPrimitivePipelines::new(&device).unwrap();
        let mut cases = Vec::new();
        for (contexts, packed, independent, batched) in [
            ([100, 300], false, false, false),
            ([100, 300], true, false, false),
            ([100, 300], true, true, false),
            ([618, 1088], true, false, true),
        ] {
            let rows = contexts.map(|position| Row {
                params: params(position),
                projection: projection(1),
            });
            let packed = packed.then(|| projection(2));
            let expected = evidence(
                &a,
                &l,
                &p,
                ElementType::F32,
                2,
                8192,
                packed,
                batched,
                independent,
                &rows,
            );
            let frozen = freeze(
                &a,
                &l,
                &p,
                ElementType::F32,
                2,
                8192,
                packed,
                batched,
                independent,
                &rows,
            );
            assert_eq!(frozen.is_some(), expected.is_some());
            if let (Some(frozen), Some(expected)) = (frozen, expected) {
                cases.push((frozen, expected));
            }
        }
        drop(a);
        drop(l);
        drop(p);
        drop(device);
        for (template, expected) in cases {
            let actual = template
                .project(&FrozenObservationInput::command(2))
                .unwrap();
            let actual = actual[0].as_ref().unwrap();
            assert_eq!(actual, &expected);
            let dispatches = actual
                .algorithm_work()
                .unwrap()
                .unwrap()
                .entries()
                .iter()
                .map(|row| row.commands())
                .sum();
            assert!(
                template.retained_payload_bytes().unwrap() <= payload_upper(dispatches, 2).unwrap()
            );
            assert_eq!(actual.algorithm_work(), expected.algorithm_work());
            assert_eq!(
                actual.independent_attention_family_v2(),
                expected.independent_attention_family_v2()
            );
            assert!(
                actual.retained_payload_bytes().unwrap()
                    + std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>()
                    <= template.projection_retained_bytes_upper_bound().unwrap()
            );
            assert!(template
                .project(&FrozenObservationInput::command(3))
                .is_err());
        }
    }
}
