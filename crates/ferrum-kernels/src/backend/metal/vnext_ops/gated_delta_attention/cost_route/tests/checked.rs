use super::*;

struct Case {
    shape: AttentionShape,
    input: Vec<PreparedLinearPart>,
    output: PreparedLinearPart,
    layout: ScratchLayout,
    staging: u64,
}
impl Case {
    fn new(
        tokens: u64,
        hidden: u64,
        heads: u64,
        format: Format,
        partitioned: bool,
        staged: bool,
    ) -> Self {
        let mut shape = shape();
        shape.hidden_size = hidden;
        shape.key_dim = heads;
        shape.value_dim = heads;
        shape.value_features = shape.value_heads * heads;
        shape.qkv_features = (2 * shape.key_heads + shape.value_heads) * heads;
        shape.qkvz_features = shape.qkv_features + shape.value_features;
        shape.qkvzba_features = shape.qkvz_features + shape.ba_features;
        let parts = if partitioned {
            vec![
                (shape.qkv_features, format),
                (shape.value_features + shape.ba_features, Format::Dense),
            ]
        } else {
            vec![(shape.qkvzba_features, format)]
        };
        let input = plain_projection(
            &resolved(&schema(&parts, hidden)),
            shape.qkvzba_features,
            hidden,
            true,
        )
        .unwrap()
        .unwrap();
        let output = plain_projection(
            &resolved(&schema(&[(hidden, Format::Q6K)], shape.value_features)),
            hidden,
            shape.value_features,
            false,
        )
        .unwrap()
        .unwrap()[0];
        let staging = if staged {
            // Numeric fixture supplies enough staging space for every input
            // partition. Actual workspace admission has separate tests.
            shape.qkvzba_features * hidden * ElementType::F16.size_bytes()
        } else {
            0
        };
        let layout = ScratchLayout::new(shape, tokens)
            .unwrap()
            .with_projection_workspace(staging)
            .unwrap();
        Self {
            shape,
            input,
            output,
            layout,
            staging,
        }
    }
    fn checked(
        &self,
        rows: &[OperationCostWorkRow],
        total: u64,
        packed: bool,
        hidden: ElementType,
        capabilities: GatedDeltaExecutionCapabilities,
        model: MetalGatedDeltaExecutionCostModel,
    ) -> Result<CheckedOperationInstance, String> {
        CheckedOperationInstance::new(
            self.shape,
            hidden,
            rows,
            total,
            packed,
            &self.input,
            self.output,
            self.layout,
            self.staging,
            capabilities,
            model,
        )
    }
    fn legacy(
        &self,
        rows: &[OperationCostWorkRow],
        total: u64,
        packed: bool,
        hidden: ElementType,
        capabilities: GatedDeltaExecutionCapabilities,
        model: MetalGatedDeltaExecutionCostModel,
    ) -> Result<OperationCostCommand, String> {
        project(
            self.shape,
            hidden,
            rows,
            total,
            packed,
            &self.input,
            self.output,
            self.layout,
            self.staging,
            capabilities,
            model,
        )
    }
    fn legacy_selection(
        &self,
        rows: &[OperationCostWorkRow],
        total: u64,
        packed: bool,
        capabilities: GatedDeltaExecutionCapabilities,
        model: MetalGatedDeltaExecutionCostModel,
    ) -> Option<LegacySelection> {
        let mut projected = Vec::new();
        projected.try_reserve_exact(rows.len()).ok()?;
        let mut start = 0_u64;
        for row in rows {
            let n = row.count.get();
            let form = self.shape.execution_form(n, capabilities, model).ok()?;
            let launches =
                project_launches(self.shape, &self.input, self.output, self.layout, start, n)
                    .ok()?;
            projected.push((self.shape.params(n).ok()?, form, launches));
            start = start.checked_add(n)?;
        }
        let packed = if packed {
            Some((
                self.shape.params(total).ok()?,
                project_launches(self.shape, &self.input, self.output, self.layout, 0, total)
                    .ok()?,
            ))
        } else {
            None
        };
        Some(LegacySelection {
            rows: projected,
            packed,
        })
    }
}
type Launches = (Vec<LinearLaunch>, LinearLaunch);
struct LegacySelection {
    rows: Vec<(GatedDeltaParams, GatedDeltaExecutionForm, Launches)>,
    packed: Option<(GatedDeltaParams, Launches)>,
}
fn assert_projection(
    actual: selected::Projection<'_>,
    params: GatedDeltaParams,
    launches: &Launches,
    staged: bool,
) {
    // Include the complete private launch value: offsets, ABI, plain/transformed
    // algorithm choices and parameters, not just dispatch counts. Formatting
    // occurs only in this CPU parity assertion, outside timing measurements.
    assert_eq!(format!("{:?}", actual.params), format!("{params:?}"));
    assert_eq!(format!("{:?}", actual.input), format!("{:?}", launches.0));
    assert_eq!(format!("{:?}", actual.output), format!("{:?}", launches.1));
    assert_eq!(actual.staged, staged);
}

#[test]
fn metal_gdn_checked_query_all_forms_and_leaf_partitions_match_legacy() {
    for counts in [
        vec![1],
        vec![1, 3],
        vec![63, 64],
        vec![64, 64],
        vec![767, 1],
        vec![768, 1],
        vec![1; 8],
    ] {
        let rows = counts.iter().copied().map(row).collect::<Vec<_>>();
        let total = counts.iter().sum();
        for (width, head_dim) in [(256, 64), (2048, 128)] {
            for format in [Format::Dense, Format::Q4K, Format::Q6K] {
                for partitioned in [false, true] {
                    for staged in [false, true] {
                        let case = Case::new(total, width, head_dim, format, partitioned, staged);
                        for packed in [false, true] {
                            for hidden in [ElementType::F16, ElementType::F32] {
                                for caps in [
                                    capabilities(),
                                    GatedDeltaExecutionCapabilities::recurrent_only(),
                                ] {
                                    for simd in [16, 32] {
                                        let model = MetalGatedDeltaExecutionCostModel::initial_c64(
                                            simd, 128,
                                        );
                                        let instance = case
                                            .checked(&rows, total, packed, hidden, caps, model)
                                            .unwrap();
                                        assert_eq!(
                                            instance.command().unwrap(),
                                            case.legacy(&rows, total, packed, hidden, caps, model)
                                                .unwrap()
                                        );
                                        let old = case
                                            .legacy_selection(&rows, total, packed, caps, model)
                                            .unwrap();
                                        let (selected_packed, selected_rows) =
                                            instance.selected_projections().unwrap();
                                        assert_eq!(selected_packed.is_some(), old.packed.is_some());
                                        if let (Some(actual), Some((params, launches))) =
                                            (selected_packed, old.packed.as_ref())
                                        {
                                            assert_projection(
                                                actual,
                                                *params,
                                                launches,
                                                case.staging != 0,
                                            );
                                        }
                                        let selected_rows = selected_rows.collect::<Vec<_>>();
                                        assert_eq!(selected_rows.len(), old.rows.len());
                                        for (actual, (params, form, launches)) in
                                            selected_rows.into_iter().zip(&old.rows)
                                        {
                                            assert_eq!(actual.form, *form);
                                            assert_projection(
                                                actual.projection,
                                                *params,
                                                launches,
                                                case.staging != 0,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn metal_gdn_checked_query_preserves_numeric_and_command_failures() {
    let case = Case::new(2, 256, 64, Format::Q4K, false, false);
    let model = MetalGatedDeltaExecutionCostModel::initial_c64(32, 128);
    for packed in [false, true] {
        for (rows, total, hidden) in [
            (vec![row(1)], 2, ElementType::F16),
            (vec![row(3)], 3, ElementType::F16),
            (vec![], 2, ElementType::F16),
            (vec![row(1)], 1, ElementType::I8),
            (
                vec![row(u64::from(u32::MAX))],
                u64::from(u32::MAX),
                ElementType::F16,
            ),
        ] {
            let old = case.legacy(&rows, total, packed, hidden, capabilities(), model);
            let current = case
                .checked(&rows, total, packed, hidden, capabilities(), model)
                .and_then(|instance| instance.command());
            assert!(old.is_err());
            assert_eq!(current, old);
        }
    }
}

#[test]
fn metal_gdn_checked_query_keeps_each_query_shape_and_launches_local() {
    let case = Case::new(128, 256, 64, Format::Q4K, false, false);
    let model = MetalGatedDeltaExecutionCostModel::initial_c64(16, 128);
    let first = case
        .checked(
            &[row(63), row(65)],
            128,
            true,
            ElementType::F16,
            capabilities(),
            model,
        )
        .unwrap();
    let first_command = first.command().unwrap();
    let next = case
        .checked(
            &[row(64), row(64)],
            128,
            true,
            ElementType::F16,
            capabilities(),
            model,
        )
        .unwrap();
    assert_ne!(
        first_command.native_operation(),
        next.command().unwrap().native_operation()
    );
    assert_eq!(first.command().unwrap(), first_command);
    let (_, first_rows) = first.selected_projections().unwrap();
    let (_, next_rows) = next.selected_projections().unwrap();
    assert_eq!(
        first_rows
            .map(|r| r.projection.params.tokens)
            .collect::<Vec<_>>(),
        [63, 65]
    );
    assert_eq!(
        next_rows
            .map(|r| r.projection.params.tokens)
            .collect::<Vec<_>>(),
        [64, 64]
    );
}

#[test]
#[ignore = "CPU diagnostic; prints both construction paths without a performance threshold"]
fn metal_gdn_checked_query_cpu_timing() {
    use std::{hint::black_box, time::Instant};
    fn consume<'a>(
        packed: Option<selected::Projection<'a>>,
        rows: impl Iterator<Item = selected::Row<'a>>,
    ) {
        black_box(packed);
        for row in rows {
            black_box(row);
        }
    }
    fn projection(
        params: GatedDeltaParams,
        launches: &Launches,
        staged: bool,
    ) -> selected::Projection<'_> {
        selected::Projection {
            params,
            input: launches.0.as_slice(),
            output: launches.1,
            staged,
        }
    }
    for counts in [vec![1], vec![1; 8], vec![63, 64]] {
        let total = counts.iter().sum();
        let rows = counts.iter().copied().map(row).collect::<Vec<_>>();
        let case = Case::new(total, 2048, 64, Format::Q4K, true, true);
        let model = MetalGatedDeltaExecutionCostModel::initial_c64(32, 128);
        for packed in [false, true] {
            let instance = case
                .checked(
                    &rows,
                    total,
                    packed,
                    ElementType::F16,
                    capabilities(),
                    model,
                )
                .unwrap();
            assert_eq!(
                instance.command().unwrap(),
                case.legacy(
                    &rows,
                    total,
                    packed,
                    ElementType::F16,
                    capabilities(),
                    model
                )
                .unwrap()
            );
            let iterations = 2000_u32;
            let old_start = Instant::now();
            for _ in 0..iterations {
                black_box(
                    case.legacy(
                        black_box(&rows),
                        total,
                        packed,
                        ElementType::F16,
                        capabilities(),
                        model,
                    )
                    .unwrap(),
                );
                let selected = case
                    .legacy_selection(black_box(&rows), total, packed, capabilities(), model)
                    .unwrap();
                consume(
                    selected
                        .packed
                        .as_ref()
                        .map(|(params, launches)| projection(*params, launches, case.staging != 0)),
                    selected
                        .rows
                        .iter()
                        .map(|(params, form, launches)| selected::Row {
                            projection: projection(*params, launches, case.staging != 0),
                            form: *form,
                        }),
                );
            }
            let old_ns = old_start.elapsed().as_nanos() / u128::from(iterations);
            let new_start = Instant::now();
            for _ in 0..iterations {
                let instance = case
                    .checked(
                        black_box(&rows),
                        total,
                        packed,
                        ElementType::F16,
                        capabilities(),
                        model,
                    )
                    .unwrap();
                black_box(instance.command().unwrap());
                let (packed, rows) = instance.selected_projections().unwrap();
                consume(packed, rows);
            }
            let new_ns = new_start.elapsed().as_nanos() / u128::from(iterations);
            eprintln!("metal_gdn_checked_query rows={} total={total} packed={packed} legacy_construct_ns={old_ns} shared_construct_ns={new_ns}; excludes common selected evidence renderer and physical projection", rows.len());
        }
    }
}
