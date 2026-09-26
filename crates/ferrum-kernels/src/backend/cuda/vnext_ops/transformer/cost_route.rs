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
    kind: Primitive,
    hidden: u64,
    capture: SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let participants =
        u32::try_from(request.rows().len()).map_err(|_| VNextError::InvalidExecutionPlan {
            reason: "CUDA cost participant count exceeds u32".into(),
        })?;
    Ok(Some(OperationCostRoute::new(vec![command(
        kind,
        participants,
        request.immediate_tokens(),
        hidden,
        capture,
    )?])?))
}

pub(super) fn rms_norm(
    request: OperationCostRouteRequest<'_>,
    precision: RmsNormPrecision,
    capture: SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let validate = || -> Result<(), String> {
        if request.operation_id().as_str() != precision.operation() {
            return Err("CUDA RMSNorm cost query selected another operation".into());
        }
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        rational_attribute(request.attributes(), "epsilon")?;
        validate_rms_norm(
            binding(request.bindings(), ResolvedValueRole::Input, 0)?,
            binding(request.bindings(), ResolvedValueRole::Input, 1)?,
            binding(request.bindings(), ResolvedValueRole::Output, 0)?,
            hidden,
            precision,
        )?;
        checked_u32(request.immediate_tokens(), "RMSNorm row count")?;
        checked_i32(hidden, "RMSNorm hidden size")?;
        Ok(())
    };
    validate().map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    if !packed_bindings(
        &request,
        &[
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Output, 0),
        ],
    )? {
        return Ok(None);
    }
    let hidden = unsigned_attribute(request.attributes(), "hidden_size")
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    let epsilon = rational_attribute(request.attributes(), "epsilon")
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    checked_route(
        &request,
        Primitive::RmsNorm { precision, epsilon },
        hidden,
        capture,
    )
}

pub(super) fn residual(
    request: OperationCostRouteRequest<'_>,
    precision: ResidualPrecision,
    capture: SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let validate = || -> Result<(), String> {
        if request.operation_id().as_str() != precision.operation() {
            return Err("CUDA residual cost query selected another operation".into());
        }
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        validate_residual_add(
            binding(request.bindings(), ResolvedValueRole::Input, 0)?,
            binding(request.bindings(), ResolvedValueRole::Input, 1)?,
            binding(request.bindings(), ResolvedValueRole::Output, 0)?,
            hidden,
            precision,
        )?;
        let elements = request
            .immediate_tokens()
            .checked_mul(hidden)
            .ok_or("CUDA residual cost element extent overflows")?;
        checked_i32(elements, "residual add element count")?;
        checked_u32(
            elements.div_ceil(u64::from(THREADS_PER_BLOCK)),
            "residual add launch grid",
        )?;
        Ok(())
    };
    validate().map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
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
    let hidden = unsigned_attribute(request.attributes(), "hidden_size")
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    checked_route(
        &request,
        Primitive::ResidualAdd { precision },
        hidden,
        capture,
    )
}

// Metadata follows the selected launch, including dtype, reduction width and
// epsilon. Numeric extents remain work, never an algorithm inferred from a hash.
pub(super) fn selected(
    kind: Primitive,
    tokens: u64,
    hidden: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, tokens)?;
    if tokens == 0 || hidden == 0 {
        return None;
    }
    let (entry, numerical, threads, epsilon, work) = match kind {
        Primitive::RmsNorm { precision, epsilon } => {
            let rows = u32::try_from(tokens).ok()?;
            let hidden_i32 = i32::try_from(hidden).ok()?;
            if !epsilon.is_finite() || epsilon < 0.0 {
                return None;
            }
            static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
            (
                precision.kernel(),
                *NUMERICAL.get_or_init(|| Sha256::digest(crate::ptx::RMS_NORM.as_bytes()).into()),
                rms_norm_threads(hidden_i32),
                epsilon.to_bits(),
                KernelNumericWorkV1 {
                    logical_units: tokens,
                    padded_units: tokens,
                    inner_units_per_logical_unit: hidden,
                    grid: [rows, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
            )
        }
        Primitive::ResidualAdd { precision } => {
            let elements = tokens.checked_mul(hidden)?;
            i32::try_from(elements).ok()?;
            let grid = u32::try_from(elements.div_ceil(u64::from(THREADS_PER_BLOCK))).ok()?;
            static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
            (
                precision.kernel(),
                *NUMERICAL
                    .get_or_init(|| Sha256::digest(crate::ptx::RESIDUAL_ADD.as_bytes()).into()),
                THREADS_PER_BLOCK,
                0,
                KernelNumericWorkV1 {
                    logical_units: elements,
                    padded_units: u64::from(grid).checked_mul(u64::from(THREADS_PER_BLOCK))?,
                    inner_units_per_logical_unit: 1,
                    grid: [grid, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
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
    builder
        .kernel_with_replay_geometry(
            algorithm,
            work,
            KernelReplayGeometryV1 {
                block: [threads, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &[hidden, u64::from(epsilon)],
            },
        )
        .ok()?;

    builder.finish().ok()
}

#[cfg(test)]
mod selected_tests {
    use super::*;
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
