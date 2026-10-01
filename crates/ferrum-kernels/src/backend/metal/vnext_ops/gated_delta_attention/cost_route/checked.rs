//! Numeric launch decisions owned by one provider query. No live resources,
//! pipeline handles, persisted prediction or replay permission are retained.
use super::*;

struct CheckedProjection {
    input: Vec<LinearLaunch>,
    output: LinearLaunch,
}

impl CheckedProjection {
    fn dispatches(
        &self,
        policy: Option<staged_prefill::StagingPolicy>,
    ) -> Result<ProjectionDispatches, String> {
        let input = self.input.iter().try_fold(0_u64, |count, &launch| {
            count
                .checked_add(staged_prefill::policy_dispatch_count(launch, policy))
                .ok_or_else(|| "Metal GDN input dispatch count overflows".to_owned())
        })?;
        Ok(ProjectionDispatches {
            input,
            output: self.output.dispatch_count(),
        })
    }

    fn selected(&self, params: GatedDeltaParams, staged: bool) -> selected::Projection<'_> {
        selected::Projection {
            params,
            input: &self.input,
            output: self.output,
            staged,
        }
    }
}

struct CheckedRow {
    params: GatedDeltaParams,
    form: GatedDeltaExecutionForm,
    // Packed row projections are evidence-only, as on the original route.
    // Their failure must not discard an otherwise valid packed command.
    projection: Option<CheckedProjection>,
}

pub(super) struct CheckedOperationInstance {
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    staged: bool,
    packed: Option<CheckedProjection>,
    packed_params: Option<GatedDeltaParams>,
    rows: Vec<CheckedRow>,
}

impl CheckedOperationInstance {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        shape: AttentionShape,
        hidden: ElementType,
        rows: &[OperationCostWorkRow],
        total: u64,
        packed: bool,
        input: &[PreparedLinearPart],
        output: PreparedLinearPart,
        layout: ScratchLayout,
        staging_bytes: u64,
        capabilities: GatedDeltaExecutionCapabilities,
        cost_model: MetalGatedDeltaExecutionCostModel,
    ) -> Result<Self, String> {
        let launches = |start, count| {
            project_launches(shape, input, output, layout, start, count)
                .map(|(input, output)| CheckedProjection { input, output })
        };
        let packed_projection = if packed {
            shape.validate_launch_extents(total)?;
            Some(launches(0, total)?)
        } else {
            None
        };
        let mut projected = Vec::new();
        projected
            .try_reserve_exact(rows.len())
            .map_err(|_| "Metal GDN row metadata capacity unavailable")?;
        let mut start = 0_u64;
        for row in rows {
            let tokens = row.count.get();
            let form = shape.execution_form(tokens, capabilities, cost_model)?;
            let projection = if packed {
                launches(start, tokens).ok()
            } else {
                Some(launches(start, tokens)?)
            };
            projected.push(CheckedRow {
                params: shape.params(tokens)?,
                form,
                projection,
            });
            start = start
                .checked_add(tokens)
                .ok_or("Metal GDN projected token count overflows")?;
        }
        if start != total {
            return Err("Metal GDN projected rows differ from total tokens".into());
        }
        Ok(Self {
            hidden,
            tokens: total,
            scratch: layout.required_bytes,
            staged: staging_bytes != 0,
            packed: packed_projection,
            packed_params: if packed {
                shape.params(total).ok()
            } else {
                None
            },
            rows: projected,
        })
    }

    pub(super) fn command(&self) -> Result<OperationCostCommand, String> {
        let policy = self
            .staged
            .then_some(staged_prefill::StagingPolicy::GatedDelta);
        let packed = self
            .packed
            .as_ref()
            .map(|projection| projection.dispatches(policy))
            .transpose()?;
        command(
            self.hidden,
            self.tokens,
            packed,
            self.rows.iter().map(|row| {
                Ok(RowDispatches {
                    projections: if packed.is_some() {
                        ProjectionDispatches {
                            input: 0,
                            output: 0,
                        }
                    } else {
                        row.projection
                            .as_ref()
                            .ok_or("Metal GDN row projection is unavailable")?
                            .dispatches(policy)?
                    },
                    delta: delta_dispatch_count(row.form, &row.params),
                    chunked: matches!(row.form, GatedDeltaExecutionForm::ChunkedScan(_)),
                })
            }),
        )
    }

    pub(super) fn selected_projections(
        &self,
    ) -> Option<(
        Option<selected::Projection<'_>>,
        impl Iterator<Item = selected::Row<'_>> + Clone,
    )> {
        if self.rows.iter().any(|row| row.projection.is_none()) {
            return None;
        }
        let packed = match &self.packed {
            Some(projection) => Some(projection.selected(self.packed_params?, self.staged)),
            None => None,
        };
        let rows = self.rows.iter().map(|row| selected::Row {
            projection: row
                .projection
                .as_ref()
                .expect("checked above")
                .selected(row.params, self.staged),
            form: row.form,
        });
        Some((packed, rows))
    }

    pub(super) fn statistics(
        &self,
        attention: &MetalGatedDeltaPipelines,
        linear: &MetalLinearPipelines,
        primitives: &MetalPrimitivePipelines,
        classes: &selected::PreparedKernelClasses,
        linear_classes: Option<&PreparedLinearProjectionClasses>,
    ) -> Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1> {
        let (packed, rows) = self.selected_projections()?;
        selected::evidence_with_prepared_classes(
            attention,
            linear,
            primitives,
            self.hidden,
            self.tokens,
            self.scratch,
            packed,
            rows,
            Some(classes),
            linear_classes,
        )
    }
}
