//! CUDA eager primitive launch metadata shared with the actual encoder.
use super::*;
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostEvidenceV1,
};
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
};
use ferrum_types::SloStructuredCostCapture;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(super) enum Primitive {
    RmsNorm {
        precision: RmsNormPrecision,
        epsilon: f32,
    },
    ResidualAdd {
        precision: ResidualPrecision,
    },
}

/// Immutable semantics and algorithm identity for the exact bound node.
/// Tokens, participant rows, physical resources and selected work are absent.
pub(super) struct PreparedPrimitiveCost {
    kind: Primitive,
    hidden: u64,
    launch: Option<PreparedPrimitiveLaunch>,
}

pub(super) fn prepare_rms_norm(
    request: ferrum_interfaces::vnext::OperationCostPreparationRequest<'_>,
    precision: RmsNormPrecision,
) -> Option<ferrum_interfaces::vnext::PreparedOperationCostData> {
    PreparedPrimitiveCost::rms_norm(
        request.operation_id().as_str(),
        request.bindings(),
        request.attributes(),
        precision,
    )
    .ok()
    .map(ferrum_interfaces::vnext::PreparedOperationCostData::new)
}

pub(super) fn prepare_residual(
    request: ferrum_interfaces::vnext::OperationCostPreparationRequest<'_>,
    precision: ResidualPrecision,
) -> Option<ferrum_interfaces::vnext::PreparedOperationCostData> {
    PreparedPrimitiveCost::residual(
        request.operation_id().as_str(),
        request.bindings(),
        request.attributes(),
        precision,
    )
    .ok()
    .map(ferrum_interfaces::vnext::PreparedOperationCostData::new)
}

impl PreparedPrimitiveCost {
    fn rms_norm(
        operation: &str,
        bindings: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        precision: RmsNormPrecision,
    ) -> Result<Self, String> {
        if operation != precision.operation() {
            return Err("CUDA RMSNorm cost query selected another operation".into());
        }
        let hidden = unsigned_attribute(attributes, "hidden_size")?;
        let epsilon = rational_attribute(attributes, "epsilon")?;
        validate_rms_norm(
            binding(bindings, ResolvedValueRole::Input, 0)?,
            binding(bindings, ResolvedValueRole::Input, 1)?,
            binding(bindings, ResolvedValueRole::Output, 0)?,
            hidden,
            precision,
        )?;
        checked_i32(hidden, "RMSNorm hidden size")?;
        let kind = Primitive::RmsNorm { precision, epsilon };
        let launch = PreparedPrimitiveLaunch::new(kind, hidden);
        Ok(Self {
            kind,
            hidden,
            launch,
        })
    }

    fn residual(
        operation: &str,
        bindings: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        precision: ResidualPrecision,
    ) -> Result<Self, String> {
        if operation != precision.operation() {
            return Err("CUDA residual cost query selected another operation".into());
        }
        let hidden = unsigned_attribute(attributes, "hidden_size")?;
        validate_residual_add(
            binding(bindings, ResolvedValueRole::Input, 0)?,
            binding(bindings, ResolvedValueRole::Input, 1)?,
            binding(bindings, ResolvedValueRole::Output, 0)?,
            hidden,
            precision,
        )?;
        let kind = Primitive::ResidualAdd { precision };
        let launch = PreparedPrimitiveLaunch::new(kind, hidden);
        Ok(Self {
            kind,
            hidden,
            launch,
        })
    }

    fn validate_work(&self, tokens: u64) -> Result<(), String> {
        match self.kind {
            Primitive::RmsNorm { .. } => {
                checked_u32(tokens, "RMSNorm row count")?;
            }
            Primitive::ResidualAdd { .. } => {
                let elements = tokens
                    .checked_mul(self.hidden)
                    .ok_or("CUDA residual cost element extent overflows")?;
                checked_i32(elements, "residual add element count")?;
                checked_u32(
                    elements.div_ceil(u64::from(THREADS_PER_BLOCK)),
                    "residual add launch grid",
                )?;
            }
        }
        Ok(())
    }
}

fn command(
    kind: Primitive,
    participants: u32,
    tokens: u64,
    hidden: u64,
    capture: SloStructuredCostCapture,
) -> Result<OperationCostCommand, VNextError> {
    let command = OperationCostCommand::new(
        match kind {
            Primitive::RmsNorm { .. } => "vnext_rms_norm",
            Primitive::ResidualAdd { .. } => "vnext_residual_add",
        },
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        participants,
        tokens,
        1,
        0,
    )?;
    Ok(match selected(kind, tokens, hidden, capture) {
        Some(evidence) => command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command),
        None => command,
    })
}

pub(super) fn attach(
    encoded: CudaDeviceCommand,
    kind: Primitive,
    participants: u32,
    tokens: u64,
    hidden: u64,
    capture: SloStructuredCostCapture,
) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
    let work = command(kind, participants, tokens, hidden, capture).map_err(contract_error)?;
    encoded
        .with_work_attribution(
            work.batching(),
            work.participant_count(),
            work.token_count(),
            work.compute_dispatch_count(),
            work.transfer_command_count(),
        )
        .map(|command| command.with_statistical_evidence(work.statistical_evidence().cloned()))
}

fn packed_bindings(
    request: &OperationCostRouteRequest<'_>,
    bindings: &[(ResolvedValueRole, u32)],
) -> Result<bool, VNextError> {
    if request.rows().len() == 1 {
        return Ok(true);
    }
    for &(role, ordinal) in bindings {
        if !request.binding_uses_packed_batch_coordinates(role, ordinal)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn checked_route(
    request: &OperationCostRouteRequest<'_>,
    prepared: &PreparedPrimitiveCost,
    capture: SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let participants =
        u32::try_from(request.rows().len()).map_err(|_| VNextError::InvalidExecutionPlan {
            reason: "CUDA cost participant count exceeds u32".into(),
        })?;
    let command = OperationCostCommand::new(
        match prepared.kind {
            Primitive::RmsNorm { .. } => "vnext_rms_norm",
            Primitive::ResidualAdd { .. } => "vnext_residual_add",
        },
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        participants,
        request.immediate_tokens(),
        1,
        0,
    )?;
    let command = match prepared
        .launch
        .as_ref()
        .and_then(|launch| launch.selected(request.immediate_tokens(), capture))
    {
        Some(evidence) => command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command),
        None => command,
    };
    Ok(Some(OperationCostRoute::new(vec![command])?))
}

pub(super) fn rms_norm(
    request: OperationCostRouteRequest<'_>,
    precision: RmsNormPrecision,
    capture: SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let fallback;
    let prepared = match request.prepared_cost_data::<PreparedPrimitiveCost>() {
        Some(prepared) => prepared,
        None => {
            fallback = PreparedPrimitiveCost::rms_norm(
                request.operation_id().as_str(),
                request.bindings(),
                request.attributes(),
                precision,
            )
            .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
            &fallback
        }
    };
    // Coordinate/range evidence belongs to this dynamic instance.
    prepared
        .validate_work(request.immediate_tokens())
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    if !packed_bindings(
        &request,
        &[
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Output, 0),
        ],
    )? {
        return Ok(None);
    }
    checked_route(&request, prepared, capture)
}

pub(super) fn residual(
    request: OperationCostRouteRequest<'_>,
    precision: ResidualPrecision,
    capture: SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let fallback;
    let prepared = match request.prepared_cost_data::<PreparedPrimitiveCost>() {
        Some(prepared) => prepared,
        None => {
            fallback = PreparedPrimitiveCost::residual(
                request.operation_id().as_str(),
                request.bindings(),
                request.attributes(),
                precision,
            )
            .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
            &fallback
        }
    };
    prepared
        .validate_work(request.immediate_tokens())
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    if !packed_bindings(
        &request,
        &[
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Output, 0),
        ],
    )? {
        return Ok(None);
    }
    checked_route(&request, prepared, capture)
}

// Prepared identity follows the same selected launch as the encoder.
// Only immutable fields enter this object; every query rebuilds numeric work.
struct PreparedPrimitiveLaunch {
    kind: Primitive,
    hidden: u64,
    algorithm: SelectedAlgorithmClassV1,
    threads: u32,
    epsilon: u32,
}

#[cfg(test)]
thread_local! {
    static PREPARED_LAUNCH_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl PreparedPrimitiveLaunch {
    fn new(kind: Primitive, hidden: u64) -> Option<Self> {
        #[cfg(test)]
        PREPARED_LAUNCH_BUILDS.with(|count| count.set(count.get() + 1));
        if hidden == 0 {
            return None;
        }
        let (entry, numerical, threads, epsilon) = match kind {
            Primitive::RmsNorm { precision, epsilon } => {
                let hidden_i32 = i32::try_from(hidden).ok()?;
                if !epsilon.is_finite() || epsilon < 0.0 {
                    return None;
                }
                static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
                (
                    precision.kernel(),
                    *NUMERICAL
                        .get_or_init(|| Sha256::digest(crate::ptx::RMS_NORM.as_bytes()).into()),
                    rms_norm_threads(hidden_i32),
                    epsilon.to_bits(),
                )
            }
            Primitive::ResidualAdd { precision } => {
                static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
                (
                    precision.kernel(),
                    *NUMERICAL
                        .get_or_init(|| Sha256::digest(crate::ptx::RESIDUAL_ADD.as_bytes()).into()),
                    THREADS_PER_BLOCK,
                    0,
                )
            }
        };
        let mut layout = Sha256::new();
        layout.update(b"cuda.contiguous_primitive_launch.v1");
        layout.update(entry.as_bytes());
        layout.update(threads.to_le_bytes());
        layout.update(epsilon.to_le_bytes());
        let algorithm =
            SelectedAlgorithmClassV1::new(entry, 1, numerical, layout.finalize().into()).ok()?;
        Some(Self {
            kind,
            hidden,
            algorithm,
            threads,
            epsilon,
        })
    }

    fn selected(
        &self,
        tokens: u64,
        capture: SloStructuredCostCapture,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        let mut builder = selected_cost::builder(capture, tokens)?;
        if tokens == 0 {
            return None;
        }
        let work = match self.kind {
            Primitive::RmsNorm { .. } => KernelNumericWorkV1 {
                logical_units: tokens,
                padded_units: tokens,
                inner_units_per_logical_unit: self.hidden,
                grid: [u32::try_from(tokens).ok()?, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
            Primitive::ResidualAdd { .. } => {
                let elements = tokens.checked_mul(self.hidden)?;
                i32::try_from(elements).ok()?;
                let grid = u32::try_from(elements.div_ceil(u64::from(THREADS_PER_BLOCK))).ok()?;
                KernelNumericWorkV1 {
                    logical_units: elements,
                    padded_units: u64::from(grid).checked_mul(u64::from(THREADS_PER_BLOCK))?,
                    inner_units_per_logical_unit: 1,
                    grid: [grid, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                }
            }
        };
        builder
            .kernel_with_replay_geometry(
                self.algorithm,
                work,
                KernelReplayGeometryV1 {
                    block: [self.threads, 1, 1],
                    dynamic_shared_bytes: 0,
                    fixed_parameters: &[self.hidden, u64::from(self.epsilon)],
                },
            )
            .ok()?;
        builder.finish().ok()
    }
}

pub(super) fn selected(
    kind: Primitive,
    tokens: u64,
    hidden: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    if capture == SloStructuredCostCapture::Disabled {
        return None;
    }
    PreparedPrimitiveLaunch::new(kind, hidden)?.selected(tokens, capture)
}

#[cfg(test)]
mod selected_tests {
    use super::*;
    #[test]
    fn prepared_primitive_reuses_identity_but_recomputes_work_and_replay_binding() {
        use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
        let capture = SloStructuredCostCapture::HostSettledV1;
        for kind in [
            Primitive::RmsNorm {
                precision: RmsNormPrecision::F16,
                epsilon: 1e-5,
            },
            Primitive::ResidualAdd {
                precision: ResidualPrecision::F32F16,
            },
        ] {
            PREPARED_LAUNCH_BUILDS.with(|count| count.set(0));
            let prepared = PreparedPrimitiveLaunch::new(kind, 257).unwrap();
            let first = prepared.selected(2, capture).unwrap();
            let independently_rebuilt = prepared.selected(2, capture).unwrap();
            assert_eq!(first, independently_rebuilt);
            assert_eq!(
                first.algorithm_work(),
                independently_rebuilt.algorithm_work()
            );
            let resident =
                SelectedReplayAlgorithmTemplateV1::from_selected(&first, 2, 1, 0).unwrap();
            resident.validate_binding(&independently_rebuilt).unwrap();
            let changed = prepared.selected(3, capture).unwrap();
            assert_eq!(first.family_signature(), changed.family_signature());
            assert_ne!(first.work(), changed.work());
            assert!(resident.validate_binding(&changed).is_err());
            assert!(prepared.selected(0, capture).is_none());
            assert!(prepared.selected(u64::MAX, capture).is_none());
            assert!(prepared
                .selected(2, SloStructuredCostCapture::Disabled)
                .is_none());
            assert_eq!(PREPARED_LAUNCH_BUILDS.with(|count| count.get()), 1);
            // The existing encoder path and prepared future path retain the
            // entire statistical table and resident replay template protocol.
            let encoded = selected(kind, 2, 257, capture).unwrap();
            assert_eq!(first, encoded);
            assert_eq!(first.algorithm_work(), encoded.algorithm_work());
            assert_eq!(
                resident,
                SelectedReplayAlgorithmTemplateV1::from_selected(&encoded, 2, 1, 0).unwrap()
            );
        }
    }
    #[test]
    fn cuda_selected_primitives_keep_precision_reduction_and_numeric_work() {
        let on = SloStructuredCostCapture::HostSettledV1;
        let rms = Primitive::RmsNorm {
            precision: RmsNormPrecision::F16,
            epsilon: 1e-5,
        };
        let a = selected(rms, 2, 33, on).unwrap();
        let b = selected(rms, 7, 35, on).unwrap();
        a.validate_command(2, 1, 0).unwrap();
        assert_eq!(a.family_signature(), b.family_signature());
        assert_eq!(a.work().inner_work_units, 66);
        assert_eq!(b.work().inner_work_units, 245);
        let template =
            ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1::from_selected(
                &a, 2, 1, 0,
            )
            .unwrap();
        template
            .validate_binding(&selected(rms, 2, 33, on).unwrap())
            .unwrap();
        // Same reduction bucket/grid/family does not update resident hidden_size.
        let changed_hidden = selected(rms, 2, 35, on).unwrap();
        assert_eq!(a.family_signature(), changed_hidden.family_signature());
        assert!(template.validate_binding(&changed_hidden).is_err());
        for other in [
            Primitive::RmsNorm {
                precision: RmsNormPrecision::F32ToF16,
                epsilon: 1e-5,
            },
            Primitive::RmsNorm {
                precision: RmsNormPrecision::F16,
                epsilon: 1e-6,
            },
        ] {
            assert_ne!(
                a.family_signature(),
                selected(other, 2, 33, on).unwrap().family_signature()
            );
        }
        assert_ne!(
            a.family_signature(),
            selected(rms, 2, 65, on).unwrap().family_signature()
        );
        let residual = Primitive::ResidualAdd {
            precision: ResidualPrecision::F32F16,
        };
        let c = command(residual, 2, 3, 257, on).unwrap();
        let e = c.statistical_evidence().unwrap();
        assert_eq!(e.work().logical_units, 771);
        assert_eq!(e.work().padded_units, 1024);
        assert_eq!(e.work().grid_blocks, 4);
        e.algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(e)
            .unwrap();
        assert!(
            command(residual, 2, 3, 257, SloStructuredCostCapture::Disabled)
                .unwrap()
                .statistical_evidence()
                .is_none()
        );
        assert!(selected(residual, u64::MAX, 257, on).is_none());
        assert!(selected(rms, 1, u64::MAX, on).is_none());
        assert!(selected(rms, 0, 33, on).is_none());
    }
}
