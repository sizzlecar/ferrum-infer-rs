//! Cold preparation and leased execution of the locked upstream projection ABI.
//! Native planning (including shared-memory configuration) never runs in capture.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{sys::CUdevice_attribute::*, CudaContext, CudaStream};
use ferrum_interfaces::vnext::{
    PreparedProjectionNumerics, PreparedUpstreamProjectionRoute, PreparedUpstreamProjectionWave,
    ProjectionBlockFormat, UpstreamNativeGeometry, UpstreamNativePlanFacts,
    UpstreamProjectionArithmetic, UpstreamProjectionLayout, UpstreamProjectionWaveFacts,
    UpstreamScratchEstimate, UpstreamScratchRole,
};

use crate::backend::cuda::vnext_runtime::CudaDeviceRuntimeError;
use crate::native_ops::upstream_linear::{
    Algorithm, Arithmetic, Device, DeviceSpan, ExtraFormat, Format, Layout, PreparedUpstreamLinear,
};

mod preparation;
pub(in crate::backend::cuda::vnext_ops) use preparation::ProjectionPreparation;
use preparation::ReplayFingerprint;

mod weight_validation;
pub(in crate::backend::cuda::vnext_ops) use weight_validation::{
    WeightValidation, WeightValidationRegistry,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NativeFamily {
    Base,
    Extra,
}

type NativeKey = (NativeFamily, u32, u32, u32, u32, u32);

/// Retained matrix validation follows this explicit profile's format family;
/// the original G32 validator remains restricted to its own three formats.
pub(in crate::backend::cuda::vnext_ops) fn validate_parts(
    profile: ferrum_interfaces::vnext::UpstreamMarkerV2Profile,
    projection: &ferrum_interfaces::vnext::PreparedProjection,
    parts: &[super::weights::MatrixPart],
) -> Result<(), String> {
    use crate::gguf_blocks::GgufBlockFormat as Block;
    let formats: &[Block] = if profile.extra() {
        &[
            Block::Q3K,
            Block::Q4K,
            Block::Q5K,
            Block::Iq3S,
            Block::Iq4Nl,
            Block::Iq4Xs,
        ]
    } else {
        &[Block::Q4K, Block::Q5K, Block::Iq4Xs]
    };
    super::q8act::Q8ActKernels::validate_parts_for_formats(formats, projection, parts)
}

fn block_format(format: ProjectionBlockFormat) -> crate::gguf_blocks::GgufBlockFormat {
    use crate::gguf_blocks::GgufBlockFormat as Block;
    match format {
        ProjectionBlockFormat::Q4K => Block::Q4K,
        ProjectionBlockFormat::Q5K => Block::Q5K,
        ProjectionBlockFormat::Iq4Xs => Block::Iq4Xs,
        ProjectionBlockFormat::Q3K => Block::Q3K,
        ProjectionBlockFormat::Iq3S => Block::Iq3S,
        ProjectionBlockFormat::Iq4Nl => Block::Iq4Nl,
    }
}

pub(in crate::backend::cuda::vnext_ops) struct UpstreamPlanFactory {
    context: Arc<CudaContext>,
    device: Device,
    fingerprint: String,
    // Geometry contains no allocation addresses or resource authorization.
    plans: Mutex<BTreeMap<NativeKey, Arc<PreparedUpstreamLinear>>>,
    waves: Mutex<BTreeMap<(String, UpstreamProjectionWaveFacts), Arc<PreparedUpstreamLeaf>>>,
}

impl UpstreamPlanFactory {
    pub fn new(context: &Arc<CudaContext>, fingerprint: String) -> Result<Self, String> {
        let attr = |key| context.attribute(key).map_err(|error| error.to_string());
        let major = u32::try_from(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?)
            .map_err(|_| "invalid CUDA major capability")?;
        let minor = u32::try_from(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?)
            .map_err(|_| "invalid CUDA minor capability")?;
        let device = Device {
            architecture: major * 100 + minor * 10,
            multiprocessors: u32::try_from(attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)
                .map_err(|_| "invalid CUDA SM count")?,
            maximum_dynamic_shared_bytes: u64::try_from(attr(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?)
            .map_err(|_| "invalid CUDA shared-memory limit")?,
        };
        if fingerprint.is_empty() {
            return Err("upstream implementation fingerprint is absent".into());
        }
        Ok(Self {
            context: context.clone(),
            device,
            fingerprint,
            plans: Mutex::new(BTreeMap::new()),
            waves: Mutex::new(BTreeMap::new()),
        })
    }

    fn native(
        &self,
        arithmetic: UpstreamProjectionArithmetic,
        format: ProjectionBlockFormat,
        rows: u32,
        inputs: u32,
        outputs: u32,
    ) -> Result<Arc<PreparedUpstreamLinear>, String> {
        let (family, algorithm) = match arithmetic {
            UpstreamProjectionArithmetic::MmqD4MarkerV2
            | UpstreamProjectionArithmetic::MmqDs4MarkerV2 => (NativeFamily::Base, Algorithm::Mmq),
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2 => (NativeFamily::Base, Algorithm::Mmvq),
            UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2 => {
                (NativeFamily::Extra, Algorithm::Mmq)
            }
            UpstreamProjectionArithmetic::MmvqQ8_1ExtraMarkerV2 => {
                (NativeFamily::Extra, Algorithm::Mmvq)
            }
            _ => return Err("this provider requires actual MarkerV2 device arithmetic".into()),
        };
        let (base, extra) = match (family, format) {
            (NativeFamily::Base, ProjectionBlockFormat::Iq4Xs) => (Some(Format::Iq4Xs), None),
            (NativeFamily::Base, ProjectionBlockFormat::Q4K) => (Some(Format::Q4K), None),
            (NativeFamily::Base, ProjectionBlockFormat::Q5K) => (Some(Format::Q5K), None),
            (NativeFamily::Extra, ProjectionBlockFormat::Q3K) => (None, Some(ExtraFormat::Q3K)),
            (NativeFamily::Extra, ProjectionBlockFormat::Iq3S) => (None, Some(ExtraFormat::Iq3S)),
            (NativeFamily::Extra, ProjectionBlockFormat::Iq4Nl) => (None, Some(ExtraFormat::Iq4Nl)),
            _ => return Err("projection arithmetic and native format family differ".into()),
        };
        let format_code = base
            .map(|format| format as u32)
            .or_else(|| extra.map(|format| format as u32))
            .ok_or("native format is absent")?;
        let key = (family, algorithm as u32, format_code, rows, inputs, outputs);
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| "upstream plan cache poisoned")?;
        if let Some(plan) = plans.get(&key) {
            return Ok(plan.clone());
        }
        self.context
            .bind_to_thread()
            .map_err(|error| error.to_string())?;
        let plan = match (base, extra) {
            (Some(format), None) => PreparedUpstreamLinear::new(
                algorithm,
                Arithmetic::MarkerV2,
                format,
                Layout::Columns,
                rows,
                inputs,
                outputs,
                self.device,
            ),
            // Only an explicitly declared extra prefill interval can select
            // these rows. Small requests keep their original constructor and
            // numerical route; an old archive cannot satisfy the v2 planner.
            (None, Some(format)) if rows > 32 && algorithm == Algorithm::Mmq => {
                PreparedUpstreamLinear::new_extra_prefill(
                    format,
                    rows,
                    inputs,
                    outputs,
                    self.device,
                )
            }
            (None, Some(format)) => PreparedUpstreamLinear::new_extra(
                algorithm,
                Arithmetic::MarkerV2,
                format,
                Layout::Columns,
                rows,
                inputs,
                outputs,
                self.device,
            ),
            _ => return Err("native format family is ambiguous".into()),
        }
        .map_err(|error| format!("{error}; family={family:?} algorithm={algorithm:?} format={format_code} rows={rows} inputs={inputs} outputs={outputs}"))?;
        let plan = Arc::new(plan);
        plans.insert(key, plan.clone());
        Ok(plan)
    }

    /// Sequential leaf scratch: exact bounded small-row plans plus a checked
    /// affine envelope for the explicit fixed-J32 prefill interval.
    pub fn maximum_scratch(
        &self,
        prepared: &PreparedProjectionNumerics,
    ) -> Result<UpstreamScratchEstimate, String> {
        let mut maximum = UpstreamScratchEstimate::default();
        for projection in prepared.projections() {
            for leaf in projection.leaves().iter().filter(|leaf| leaf.is_staged()) {
                let ferrum_interfaces::vnext::WeightEncoding::BlockQuantized(spec) =
                    leaf.encoding()
                else {
                    return Err("staged upstream leaf is not block quantized".into());
                };
                let k = projection.input_features();
                let n = leaf.output_features();
                let weight_bytes = n
                    .checked_mul(k / u64::from(spec.logical_values_per_block))
                    .and_then(|blocks| blocks.checked_mul(u64::from(spec.bytes_per_block)))
                    .ok_or("upstream weight extent overflows")?;
                let declaration = prepared
                    .contract()
                    .projections
                    .iter()
                    .find(|value| value.role == projection.role())
                    .ok_or("missing projection declaration")?;
                for declared in &declaration.leaves {
                    let expected = block_format(declared.format);
                    if crate::gguf_blocks::GgufBlockFormat::from_spec(spec).ok() != Some(expected) {
                        continue;
                    }
                    let policy = declared
                        .arithmetic
                        .upstream_policy()
                        .ok_or("missing native projection policy")?;
                    for route in &policy.routes {
                        if let Some(range) = route.prefill_rows {
                            // One device qualification plan per geometry, never one per row.
                            let plan = self.native(
                                route.arithmetic,
                                policy.format,
                                range.last,
                                u32::try_from(k).map_err(|_| "K exceeds native ABI")?,
                                u32::try_from(n).map_err(|_| "N exceeds native ABI")?,
                            )?;
                            if plan.geometry().j != 32 {
                                return Err("prefill requires fixed J32".into());
                            }
                            maximum.include(UpstreamScratchEstimate::mmq_prefill(
                                k,
                                n,
                                self.device.multiprocessors,
                            )?);
                        }
                        for &rows in &route.local_rows {
                            let bytes = |width: u64| {
                                u64::from(rows)
                                    .checked_mul(width)
                                    .and_then(|value| value.checked_mul(2))
                                    .ok_or("upstream F16 span overflows")
                            };
                            let facts = UpstreamProjectionWaveFacts {
                                role: projection.role(),
                                component_id: leaf.component_id().clone(),
                                local_rows: rows,
                                layout: route.layout,
                                input_stride: k,
                                output_stride: projection.output_features(),
                                input_byte_offset: 0,
                                output_byte_offset: 0,
                                weight_byte_offset: 0,
                                input_available_bytes: bytes(k)?,
                                output_available_bytes: bytes(projection.output_features())?,
                                weight_available_bytes: weight_bytes,
                                // This provider does not create new padded weights.
                                retained_zero_padded_weight_rows: n,
                            };
                            maximum.fixed_bytes = maximum
                                .fixed_bytes
                                .max(self.prepare(prepared, &facts)?.scratch_bytes());
                        }
                    }
                }
            }
        }
        Ok(maximum)
    }

    /// `prepared` has already been rebuilt from the actual static bindings.
    /// Facts come from the current live views; an adjacent leaf cannot supply
    /// tail padding. Strict decisions are recorded before invoking any kernel.
    pub fn prepare(
        &self,
        prepared: &PreparedProjectionNumerics,
        facts: &UpstreamProjectionWaveFacts,
    ) -> Result<Arc<PreparedUpstreamLeaf>, String> {
        self.prepare_for(prepared, facts, ProjectionPreparation::Full)
    }

    pub fn prepare_for(
        &self,
        prepared: &PreparedProjectionNumerics,
        facts: &UpstreamProjectionWaveFacts,
        purpose: ProjectionPreparation,
    ) -> Result<Arc<PreparedUpstreamLeaf>, String> {
        let key = (prepared.fingerprint().to_owned(), facts.clone());
        let mut waves = self
            .waves
            .lock()
            .map_err(|_| "upstream wave cache poisoned")?;
        if let Some(wave) = waves.get(&key) {
            return Ok(wave.clone());
        }
        let wave = Arc::new(self.prepare_uncached(prepared, facts, purpose)?);
        // Arbitrary strict prefill lengths do not create a permanent cache entry.
        if wave.native.is_some()
            || matches!(
                wave.decision.route(),
                PreparedUpstreamProjectionRoute::G32 { .. }
            )
        {
            waves.insert(key, wave.clone());
        }
        Ok(wave)
    }

    fn prepare_uncached(
        &self,
        prepared: &PreparedProjectionNumerics,
        facts: &UpstreamProjectionWaveFacts,
        purpose: ProjectionPreparation,
    ) -> Result<PreparedUpstreamLeaf, String> {
        let projection = prepared
            .projection(facts.role)
            .ok_or("missing projection role")?;
        let leaf = projection
            .leaves()
            .iter()
            .find(|leaf| leaf.component_id() == &facts.component_id)
            .ok_or("missing projection component")?;
        let declaration = prepared
            .contract()
            .projections
            .iter()
            .find(|declaration| declaration.role == facts.role)
            .ok_or("missing projection declaration")?;
        let policy = declaration.leaves.iter().find_map(|candidate| {
            let ferrum_interfaces::vnext::WeightEncoding::BlockQuantized(spec) = leaf.encoding()
            else {
                return None;
            };
            let expected = block_format(candidate.format);
            if crate::gguf_blocks::GgufBlockFormat::from_spec(spec).ok() != Some(expected) {
                return None;
            }
            candidate.arithmetic.upstream_policy()
        });
        let selection = policy.and_then(|policy| {
            policy
                .select_arithmetic(
                    facts.layout,
                    facts.local_rows,
                    projection.input_features(),
                    leaf.output_features(),
                )
                .map(|arithmetic| (policy.format, arithmetic))
        });
        let native = if leaf.is_staged() && facts.weight_byte_offset % 4 == 0 {
            selection
                .map(|(format, arithmetic)| {
                    if facts.layout != UpstreamProjectionLayout::Columns {
                        return Err("Columns provider received a different declared layout".into());
                    }
                    self.native(
                        arithmetic,
                        format,
                        facts.local_rows,
                        u32::try_from(projection.input_features())
                            .map_err(|_| "K exceeds native ABI")?,
                        u32::try_from(leaf.output_features())
                            .map_err(|_| "N exceeds native ABI")?,
                    )
                })
                .transpose()?
        } else {
            None
        };
        let native_facts = native.as_ref().map(|plan| self.facts(plan));
        let (decision, fingerprint) =
            purpose.prepare_wave(prepared, facts, native_facts.as_ref())?;
        match decision.route() {
            PreparedUpstreamProjectionRoute::G32 { format, .. } => {
                tracing::debug!(target: "ferrum::cuda::upstream_projection", event = "prepared_projection_route", role = ?facts.role, component = facts.component_id.as_str(), local_rows = facts.local_rows, route = "g32", format = ?format, "selected exact declared G32 arithmetic")
            }
            PreparedUpstreamProjectionRoute::StrictBase { reason } => tracing::debug!(
                target: "ferrum::cuda::upstream_projection",
                event = "prepared_projection_route", role = ?facts.role,
                component = facts.component_id.as_str(), local_rows = facts.local_rows,
                inputs = projection.input_features(), outputs = leaf.output_features(),
                route = "strict_base", reason = ?reason,
                "selected explicit strict projection fallback"
            ),
            PreparedUpstreamProjectionRoute::Selected { arithmetic, .. } => tracing::debug!(
                target: "ferrum::cuda::upstream_projection",
                event = "prepared_projection_route", role = ?facts.role,
                component = facts.component_id.as_str(), local_rows = facts.local_rows,
                inputs = projection.input_features(), outputs = leaf.output_features(),
                route = "upstream", arithmetic = ?arithmetic,
                "selected declared upstream projection arithmetic"
            ),
        }
        let native = matches!(
            decision.route(),
            PreparedUpstreamProjectionRoute::Selected { .. }
        )
        .then_some(native)
        .flatten();
        Ok(PreparedUpstreamLeaf {
            fingerprint,
            decision,
            native,
            output_offset: leaf.output_offset(),
        })
    }

    fn facts(&self, plan: &PreparedUpstreamLinear) -> UpstreamNativePlanFacts {
        let p = plan.geometry();
        let geometry = if p.algorithm == Algorithm::Mmq as u32 {
            UpstreamNativeGeometry::Mmq {
                padded_inputs: p.padded_inputs,
                row_tile: p.j,
                column_tile: p.i,
                threads: p.nthreads,
                shared_bytes: u64::from(p.shared_bytes),
                packed_guard_blocks: p.guard_blocks,
                blocks: p.blocks,
                fixup: p.fixup != 0,
            }
        } else {
            UpstreamNativeGeometry::Mmvq {
                padded_inputs: p.padded_inputs,
                padded_outputs: p.padded_outputs,
                columns: p.ncols,
                channels: p.channels,
                warps: p.nwarps,
                rows_per_block: p.rows_per_block,
            }
        };
        UpstreamNativePlanFacts {
            implementation_fingerprint: self.fingerprint.clone(),
            device_architecture: self.device.architecture,
            multiprocessors: self.device.multiprocessors,
            maximum_dynamic_shared_bytes: self.device.maximum_dynamic_shared_bytes,
            geometry,
        }
    }
}

pub(in crate::backend::cuda::vnext_ops) struct PreparedUpstreamLeaf {
    pub decision: PreparedUpstreamProjectionWave,
    pub native: Option<Arc<PreparedUpstreamLinear>>,
    output_offset: u64,
    fingerprint: ReplayFingerprint,
}

impl PreparedUpstreamLeaf {
    pub fn replay_fingerprint(&self) -> Result<&str, String> {
        self.fingerprint.required()
    }

    pub fn scratch_bytes(&self) -> u64 {
        match self.decision.route() {
            PreparedUpstreamProjectionRoute::Selected { scratch_bytes, .. } => *scratch_bytes,
            PreparedUpstreamProjectionRoute::StrictBase { .. }
            | PreparedUpstreamProjectionRoute::G32 { .. } => 0,
        }
    }

    fn scratch(&self, region: DeviceSpan, role: UpstreamScratchRole) -> Result<DeviceSpan, String> {
        if region.bytes < self.scratch_bytes() || region.address % 16 != 0 {
            return Err("upstream scratch exceeds its admitted region or alignment".into());
        }
        let PreparedUpstreamProjectionRoute::Selected { scratch, .. } = self.decision.route()
        else {
            return Err("strict projection does not own upstream scratch".into());
        };
        let Some(part) = scratch.iter().find(|part| part.role == role) else {
            return Ok(DeviceSpan {
                address: 0,
                bytes: 0,
            });
        };
        Ok(DeviceSpan {
            address: region
                .address
                .checked_add(part.byte_offset)
                .ok_or("scratch address overflows")?,
            bytes: part.byte_len,
        })
    }

    /// Enqueue the complete inclusive path. The enclosing command owns all
    /// regions through completion; the prelude has ordered the real static
    /// coefficient scan before this cast. This method does not make allocations.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        input: DeviceSpan,
        weights: DeviceSpan,
        output: DeviceSpan,
        workspace: DeviceSpan,
        retained_weight_flag: DeviceSpan,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let error = |e: String| CudaDeviceRuntimeError::contract(e);
        let native = self
            .native
            .as_ref()
            .ok_or_else(|| error("strict route has no upstream launch".into()))?;
        let facts = self.decision.facts();
        let converted = self
            .scratch(workspace, UpstreamScratchRole::ConvertedInput)
            .map_err(error)?;
        let packed = self
            .scratch(workspace, UpstreamScratchRole::PackedActivations)
            .map_err(error)?;
        let temporary = self
            .scratch(workspace, UpstreamScratchRole::ProjectionOutput)
            .map_err(error)?;
        let fixup = self
            .scratch(workspace, UpstreamScratchRole::StreamKFixup)
            .map_err(error)?;
        let rows = self
            .scratch(workspace, UpstreamScratchRole::RowPoisonFlags)
            .map_err(error)?;
        let input_stride = u32::try_from(facts.input_stride)
            .map_err(|_| error("input stride exceeds u32".into()))?;
        let output_stride = u32::try_from(facts.output_stride)
            .map_err(|_| error("output stride exceeds u32".into()))?;
        let output_offset = self
            .output_offset
            .checked_mul(2)
            .ok_or_else(|| error("projection output offset overflows".into()))?;
        let output = DeviceSpan {
            address: output
                .address
                .checked_add(output_offset)
                .ok_or_else(|| error("projection output address overflows".into()))?,
            bytes: output
                .bytes
                .checked_sub(output_offset)
                .ok_or_else(|| error("projection output does not own its leaf offset".into()))?,
        };
        unsafe {
            native
                .pack(
                    input,
                    input_stride,
                    converted,
                    packed,
                    rows,
                    stream.cu_stream().cast(),
                )
                .map_err(|e| error(e.to_string()))?;
            native
                .dot(weights, packed, temporary, fixup, stream.cu_stream().cast())
                .map_err(|e| error(e.to_string()))?;
            native
                .cast(
                    temporary,
                    output,
                    output_stride,
                    rows,
                    retained_weight_flag,
                    stream.cu_stream().cast(),
                )
                .map_err(|e| error(e.to_string()))?;
        }
        Ok(())
    }
}
