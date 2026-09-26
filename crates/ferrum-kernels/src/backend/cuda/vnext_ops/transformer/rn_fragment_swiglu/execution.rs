use super::*;

pub(super) struct Prepared {
    pub(super) shape: Shape,
    participants: u32,
    regions: Vec<CudaBufferRegion>,
}
pub(super) fn prepare(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Prepared, String> {
    ensure_invocation(
        invocation,
        DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID,
    )?;
    if invocation.operation().version != ContractVersion::new(1, 0) {
        return Err("RN fragment FFN requires contract version 1.0".into());
    }
    let first = &invocation.participants()[0];
    let shape = Shape::from_values(
        first.bindings(),
        first.attributes(),
        invocation.work_shape().immediate_tokens(),
    )?;
    for participant in &invocation.participants()[1..] {
        let other = Shape::from_values(
            participant.bindings(),
            participant.attributes(),
            shape.tokens,
        )?;
        if other.gate_weight != shape.gate_weight || other.down_weight != shape.down_weight {
            return Err("RN fragment participant dimensions or source formats disagree".into());
        }
    }
    let regions = vec![
        shared_token_region(
            invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            shape.tokens,
        )?,
        weights::shared(invocation, 1, shape.gate_weight, shape.fragment())?,
        weights::shared(invocation, 2, shape.down_weight, shape.fragment())?,
        shared_token_region(
            invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            shape.tokens,
        )?,
        shared_scratch_region(invocation, shape.scratch_bytes)?,
    ];
    Ok(Prepared {
        shape,
        regions,
        participants: checked_u32(
            invocation.participants().len() as u64,
            "RN fragment participant count",
        )?,
    })
}

pub(super) fn encode(
    fingerprint: &str,
    mma: &CudaFunction,
    silu: &CudaFunction,
    capture: SloStructuredCostCapture,
    identity: Option<CublasHandleApiIdentity>,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    let Prepared {
        shape,
        participants,
        regions,
    } = prepare(&invocation)?;
    let selected = shape.selected(capture, identity);
    let recipe = if capture.is_disabled() {
        None
    } else {
        replay_cost::CudaReplayCostRecipe::rn_fragment(
            &invocation,
            shape,
            if shape.fragment() { None } else { identity },
        )
    };
    let mut key = CudaCommandReplayKeyBuilder::new(fingerprint, LABEL)
        .i32(shape.rows)
        .i32(shape.hidden)
        .i32(shape.intermediate)
        .u64(shape.gate_up_bytes)
        .u64(shape.scratch_bytes)
        .u32(u32::from(shape.fragment()));
    for plan in [shape.gate_weight, shape.down_weight] {
        key = key
            .u32(plan.packing_abi())
            .u32(plan::format_code(plan.source_format()))
            .u64(plan.n())
            .u64(plan.k())
            .u64(plan.packed_bytes());
    }
    let key = key.finish();
    let silu = silu.clone();
    let command = if shape.fragment() {
        let mma = mma.clone();
        CudaDeviceCommand::replayable_operation(LABEL, regions, key, move |stream, regions| {
            let (gate, activation) = scratch_addresses(shape, regions)?;
            plan::launch(
                stream,
                &mma,
                shape,
                shape.gate_weight,
                regions[0].device_ptr(),
                regions[1].device_ptr(),
                gate,
            )?;
            launch_silu_mul(
                stream,
                &silu,
                gate,
                activation,
                shape.intermediate,
                shape.activation_elements,
            )?;
            plan::launch(
                stream,
                &mma,
                shape,
                shape.down_weight,
                activation,
                regions[2].device_ptr(),
                regions[3].device_ptr(),
            )
        })
    } else {
        CudaDeviceCommand::replayable_operation_with_blas(
            LABEL,
            regions,
            key,
            move |stream, blas, regions| {
                let (gate, activation) = scratch_addresses(shape, regions)?;
                shape.gate.launch(
                    blas,
                    regions[0].device_ptr(),
                    regions[1].device_ptr(),
                    gate,
                    "RN fragment dense gate/up fallback",
                )?;
                launch_silu_mul(
                    stream,
                    &silu,
                    gate,
                    activation,
                    shape.intermediate,
                    shape.activation_elements,
                )?;
                shape.down.launch(
                    blas,
                    activation,
                    regions[2].device_ptr(),
                    regions[3].device_ptr(),
                    "RN fragment dense down fallback",
                )
            },
        )
    }
    .and_then(|command| {
        command.with_work_attribution(DeviceBatchingForm::Packed, participants, shape.tokens, 3, 0)
    })
    .map_err(|e| e.to_string())?;
    let command = command
        .with_statistical_evidence(selected)
        .with_replay_cost_recipe(recipe);
    Ok(if shape.fragment() {
        command
    } else {
        command.with_cublas_cost_requirement(identity)
    })
}
fn scratch_addresses(
    shape: Shape,
    regions: &[CudaBufferRegion],
) -> Result<(u64, u64), CudaDeviceRuntimeError> {
    let scratch = &regions[4];
    if scratch.length_bytes() < shape.scratch_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "RN fragment scratch is smaller than admitted",
        ));
    }
    let gate = scratch.device_ptr();
    let activation = gate.checked_add(shape.gate_up_bytes).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("RN fragment activation pointer overflows")
    })?;
    Ok((gate, activation))
}
