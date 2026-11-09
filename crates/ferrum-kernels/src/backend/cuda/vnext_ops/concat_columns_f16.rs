//! Exact row transport using the existing CUDA device-to-device copy path.
use super::*;

const OP: &str = ferrum_runtime::CONCAT_COLUMNS_F16_OPERATION_ID;

pub(super) struct CudaConcatColumnsProvider {
    descriptor: OperationProviderDescriptor,
}
impl CudaConcatColumnsProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = ferrum_runtime::concat_columns_f16_contract().map_err(contract_error)?;
        let cap = CapabilityId::new(ferrum_runtime::CONCAT_COLUMNS_F16_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&cap) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA concat columns capability is absent",
            ));
        }
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("concat_columns_f16.rs"),
            include_bytes!("../vnext_ops.rs"),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new("provider.cuda.concat_columns.f16").map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            fingerprint.clone(),
            ferrum_runtime::ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([cap]),
            BTreeSet::new(),
            BTreeSet::new(),
            transformer::contiguous_bindings(2),
            "resource-estimator.cuda.concat_columns.f16",
            ContractVersion::new(1, 0),
            fingerprint,
        )
        .map_err(contract_error)?
        .with_candidate_state_effects(
            ferrum_runtime::ProviderCandidateStateEffects::PrefixCausal { ports: Vec::new() },
        );
        Ok(Self { descriptor })
    }
}
impl OperationResourceEstimator for CudaConcatColumnsProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        r: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        transformer::ensure_estimator_request(&self.descriptor, &r, OP)?;
        widths(r.attributes()).map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
        Ok(transformer::estimate(
            &self.descriptor,
            r.input_fingerprint(),
            None,
        ))
    }
}
impl OperationProvider<CudaDeviceRuntime> for CudaConcatColumnsProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        encode(&i)
            .map(EncodedDeviceOperation::compute)
            .map_err(|e| provider_failure(identity, "cuda.concat_columns.encode", e))
    }
}

fn widths(a: &std::collections::BTreeMap<AttributeId, SemanticValue>) -> Result<[u64; 3], String> {
    let l = unsigned_attribute(a, "left_features")?;
    let r = unsigned_attribute(a, "right_features")?;
    let o = unsigned_attribute(a, "out_features")?;
    checked_widths(l, r, o)
}
fn checked_widths(l: u64, r: u64, o: u64) -> Result<[u64; 3], String> {
    if l == 0 || r == 0 || l.checked_add(r) != Some(o) || o > isize::MAX as u64 / 2 {
        return Err("concat columns requires positive checked left+right=output widths".into());
    }
    Ok([l * 2, r * 2, o * 2])
}
#[derive(Clone, Copy)]
struct CopyRows {
    left: u64,
    right: u64,
    output: u64,
    rows: u64,
}

fn validate_spans(
    reads: &[std::ops::Range<u64>],
    writes: &[std::ops::Range<u64>],
) -> Result<(), String> {
    for (index, write) in writes.iter().enumerate() {
        if reads
            .iter()
            .chain(&writes[..index])
            .any(|r| r.start < write.end && write.start < r.end)
        {
            return Err("concat columns output aliases another read/write in selected wave".into());
        }
    }
    Ok(())
}

fn encode(
    i: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if i.operation().id.as_str() != OP
        || i.participants().is_empty()
        || i.participant_token_ranges().len() != i.participants().len()
    {
        return Err("concat columns requires a nonempty selected token wave".into());
    }
    let first = &i.participants()[0];
    let bytes = widths(first.attributes())?;
    let mut regions = Vec::new();
    let mut copies = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for (p, range) in i.participants().iter().zip(i.participant_token_ranges()) {
        if p.attributes() != first.attributes() {
            return Err("concat columns participant attributes differ".into());
        }
        let rows = range.immediate_tokens();
        if rows == 0 {
            return Err("concat columns requires nonempty token rows".into());
        }
        let base = regions.len();
        for (column, (role, ordinal)) in [
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Output, 0),
        ]
        .into_iter()
        .enumerate()
        {
            let b = binding(p.bindings(), role, ordinal)?;
            let dims = b.tensor().dimensions();
            if dims.len() != 2
                || dims[1] != bytes[column] / 2
                || b.tensor().element_type() != ElementType::F16
                || b.tensor().layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err(
                    "concat columns resolved shape/dtype/layout differs from contract".into(),
                );
            }
            let selected = if transformer::token_binding_is_packed(i, role, ordinal)? {
                range.immediate_token_range()
            } else {
                range.source_token_range()
            };
            if selected.end > dims[0] || selected.end.checked_sub(selected.start) != Some(rows) {
                return Err("concat columns token range is invalid".into());
            }
            let r = contiguous_token_region(p, b, ElementType::F16, selected.start, rows)?;
            let expected = rows
                .checked_mul(bytes[column])
                .filter(|n| *n <= isize::MAX as u64)
                .ok_or("concat columns extent overflows")?;
            if r.length_bytes() != expected {
                return Err("concat columns physical extent differs from token rows".into());
            }
            let end = r
                .device_ptr()
                .checked_add(expected)
                .ok_or("concat columns pointer overflows")?;
            if r.device_ptr() == 0 {
                return Err("concat columns null pointer".into());
            }
            let span = r.device_ptr()..end;
            if role == ResolvedValueRole::Output {
                writes.push(span);
            } else {
                reads.push(span);
            }
            regions.push(r);
        }
        copies.push(CopyRows {
            left: regions[base].device_ptr(),
            right: regions[base + 1].device_ptr(),
            output: regions[base + 2].device_ptr(),
            rows,
        });
    }
    validate_spans(&reads, &writes)?;
    let participants =
        u32::try_from(copies.len()).map_err(|_| "concat participant count overflows")?;
    let tokens = i.work_shape().immediate_tokens();
    let transfers = tokens
        .checked_mul(2)
        .ok_or("concat transfer count overflows")?;
    CudaDeviceCommand::operation(
        "vnext_concat_columns_f16",
        regions,
        move |stream, _regions| copy_rows(stream, bytes, &copies),
    )
    .and_then(|c| {
        c.with_work_attribution(
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            tokens,
            0,
            transfers,
        )
    })
    .map_err(|e| e.to_string())
}

fn copy_rows(
    stream: &cudarc::driver::CudaStream,
    bytes: [u64; 3],
    copies: &[CopyRows],
) -> Result<(), CudaDeviceRuntimeError> {
    for p in copies {
        for row in 0..p.rows {
            let output = p.output + row * bytes[2];
            copy_last_token_rows_f16(
                stream,
                output,
                bytes[0] as usize,
                std::iter::once(p.left + row * bytes[0]),
            )?;
            copy_last_token_rows_f16(
                stream,
                output + bytes[0],
                bytes[1] as usize,
                std::iter::once(p.right + row * bytes[1]),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
