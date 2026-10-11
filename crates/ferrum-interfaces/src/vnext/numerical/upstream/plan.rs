use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::super::{
    NumericalArithmeticStage, PreparedProjectionNumerics, PreparedProjectionRoute,
    ProjectionBlockFormat, ProjectionRole, StrictProjectionReason,
    COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM,
};
use super::{UpstreamActivationPack, UpstreamProjectionArithmetic, UpstreamProjectionLayout};
use crate::vnext::WeightId;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamProjectionWaveFacts {
    pub role: ProjectionRole,
    pub component_id: WeightId,
    pub local_rows: u32,
    pub layout: UpstreamProjectionLayout,
    pub input_stride: u64,
    pub output_stride: u64,
    /// Offsets in the actual retained backing, not a logical zero-based slice.
    pub input_byte_offset: u64,
    pub output_byte_offset: u64,
    pub weight_byte_offset: u64,
    /// Bytes available *starting at* those offsets. The provider must derive
    /// these from live authorized regions, not from neighbouring allocations.
    pub input_available_bytes: u64,
    pub output_available_bytes: u64,
    pub weight_available_bytes: u64,
    /// Logical N plus proven zero-padded rows belonging to this exact leaf.
    /// Adjacent composite components are never padding authorization.
    pub retained_zero_padded_weight_rows: u64,
}

/// Native geometry comes from the pinned provider's planner, not deserialized
/// user input. This interface checks memory/ABI bounds; it does not duplicate
/// architecture-specific tile selection. Revalidation requires that planner's
/// freshly reconstructed geometry and implementation identity again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UpstreamNativeGeometry {
    Mmq {
        padded_inputs: u32,
        row_tile: u32,
        column_tile: u32,
        threads: u32,
        shared_bytes: u64,
        packed_guard_blocks: u32,
        blocks: u32,
        fixup: bool,
    },
    Mmvq {
        padded_inputs: u32,
        padded_outputs: u32,
        columns: u32,
        channels: u32,
        warps: u32,
        rows_per_block: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamNativePlanFacts {
    pub implementation_fingerprint: String,
    /// Native architecture convention, e.g. SM80=800, SM120=1200. The native
    /// planner must reject unsupported architecture tables itself as well.
    pub device_architecture: u32,
    pub multiprocessors: u32,
    pub geometry: UpstreamNativeGeometry,
    pub maximum_dynamic_shared_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamStrictReason {
    Static(StrictProjectionReason),
    RowsOrLayoutNotDeclared,
    WeightAlignmentNotSupported,
    ActivationStrideNotSupported,
    RetainedWeightPaddingNotProven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamScratchRole {
    ConvertedInput,
    PackedActivations,
    ProjectionOutput,
    StreamKFixup,
    /// Cleared and populated by each V2 pack, including replay. It must not be
    /// shared with another concurrently live input pack or a retained flag.
    RowPoisonFlags,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamRetainedWeightValidation {
    pub component_id: WeightId,
    pub weight_byte_offset: u64,
    /// Q4/Q5 MMQ half-rounded coefficients have a different finite domain
    /// from MMVQ coefficients; validation flags must never cross this identity.
    pub arithmetic: UpstreamProjectionArithmetic,
    pub format: ProjectionBlockFormat,
    pub input_features: u64,
    pub output_features: u64,
    pub byte_len: u64,
    pub alignment: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamScratchRegion {
    pub role: UpstreamScratchRole,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedUpstreamProjectionRoute {
    /// The exact old G32 stages; no upstream padding or MarkerV2 poison rules.
    G32 {
        format: ProjectionBlockFormat,
        arithmetic: super::super::StagedNumericalArithmetic,
    },
    StrictBase {
        reason: UpstreamStrictReason,
    },
    Selected {
        format: ProjectionBlockFormat,
        arithmetic: UpstreamProjectionArithmetic,
        pack: UpstreamActivationPack,
        native: UpstreamNativePlanFacts,
        scratch: Vec<UpstreamScratchRegion>,
        scratch_bytes: u64,
        /// Separately retained/admitted storage, initialized by actual GPU
        /// coefficient validation before first use. Bind to the exact weight
        /// backing and generation; never alias transient row-poison scratch.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retained_weight_validation: Option<UpstreamRetainedWeightValidation>,
    },
}

/// Static per-wave evidence only. It grants no resource lease, does not attest
/// execution, and is not trusted after deserialization until reconstructed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedUpstreamProjectionWave {
    prepared_numerics: PreparedProjectionNumerics,
    facts: UpstreamProjectionWaveFacts,
    route: PreparedUpstreamProjectionRoute,
}

impl PreparedUpstreamProjectionWave {
    pub fn route(&self) -> &PreparedUpstreamProjectionRoute {
        &self.route
    }
    pub fn facts(&self) -> &UpstreamProjectionWaveFacts {
        &self.facts
    }

    /// Call only after PreparedProjectionNumerics::validate_bindings against
    /// real retained weights. `native` must be rebuilt for these exact facts;
    /// absence for an eligible route is an error, never implicit strict fallback.
    pub fn prepare(
        prepared: &PreparedProjectionNumerics,
        facts: &UpstreamProjectionWaveFacts,
        native: Option<&UpstreamNativePlanFacts>,
    ) -> Result<Self, String> {
        prepared.validate_static_contract()?;
        let contract = prepared.contract();
        if !matches!(
            contract.schema_version,
            COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM
                | super::super::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ
                | super::super::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA
                | super::super::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
        ) {
            return Err(
                "per-wave upstream planning requires the explicit upstream composite".into(),
            );
        }
        let projection = prepared
            .projection(facts.role)
            .ok_or("missing prepared projection role")?;
        let leaf = projection
            .leaves()
            .iter()
            .find(|leaf| leaf.component_id() == &facts.component_id)
            .ok_or("wave component is not a retained projection leaf")?;
        let (m, k, n) = (
            u64::from(facts.local_rows),
            projection.input_features(),
            leaf.output_features(),
        );
        if m == 0
            || k == 0
            || n == 0
            || facts.input_stride < k
            || facts.output_stride < add(leaf.output_offset(), n)?
            || facts.input_byte_offset % 2 != 0
            || facts.output_byte_offset % 2 != 0
        {
            return Err("invalid F16 wave dimensions, stride or alignment".into());
        }
        let input_span = mul(add(mul(m - 1, facts.input_stride)?, k)?, 2)?;
        let output_span = mul(
            add(
                mul(m - 1, facts.output_stride)?,
                add(leaf.output_offset(), n)?,
            )?,
            2,
        )?;
        check_span(
            facts.input_byte_offset,
            input_span,
            facts.input_available_bytes,
        )?;
        check_span(
            facts.output_byte_offset,
            output_span,
            facts.output_available_bytes,
        )?;
        let strict = |reason| {
            Ok(Self {
                prepared_numerics: prepared.clone(),
                facts: facts.clone(),
                route: PreparedUpstreamProjectionRoute::StrictBase { reason },
            })
        };
        if let PreparedProjectionRoute::StrictBase { reason } = leaf.route() {
            return strict(UpstreamStrictReason::Static(*reason));
        }
        let override_ = contract
            .projections
            .iter()
            .find(|p| p.role == facts.role)
            .ok_or("missing projection declaration")?;
        let declared = override_.leaves.iter().find(|candidate| {
            matches!(leaf.encoding(), crate::vnext::WeightEncoding::BlockQuantized(block) if block.format_id.as_str() == candidate.format.abi().0)
        }).ok_or("staged leaf does not match its declared format")?;
        if let Some(policy) = declared.arithmetic.g32_mmq_policy() {
            if matches!(
                super::super::G32MmqPrefillPolicy::select(facts.local_rows),
                super::super::G32MmqPrefillRoute::G32
            ) && facts.layout == UpstreamProjectionLayout::Columns
            {
                if facts.input_stride != k {
                    return strict(UpstreamStrictReason::ActivationStrideNotSupported);
                }
                let weight_bytes = mul(mul(n, k / 256)?, u64::from(policy.format.abi().2))?;
                check_span(
                    facts.weight_byte_offset,
                    weight_bytes,
                    facts.weight_available_bytes,
                )?;
                if native.is_some() {
                    return Err("G32 wave cannot carry an MMQ native plan".into());
                }
                return Ok(Self {
                    prepared_numerics: prepared.clone(),
                    facts: facts.clone(),
                    route: PreparedUpstreamProjectionRoute::G32 {
                        format: policy.format,
                        arithmetic: policy.g32.clone(),
                    },
                });
            }
        }
        let policy = declared
            .arithmetic
            .upstream_policy()
            .ok_or("missing upstream leaf policy")?;
        let Some(selection) = policy
            .routes
            .iter()
            .find(|r| r.layout == facts.layout && r.contains_rows(facts.local_rows))
        else {
            return strict(UpstreamStrictReason::RowsOrLayoutNotDeclared);
        };
        if facts.weight_byte_offset % 4 != 0 {
            return strict(UpstreamStrictReason::WeightAlignmentNotSupported);
        }
        let weight_blocks_per_row = k / u64::from(policy.format.abi().1);
        let weight_row_bytes = mul(weight_blocks_per_row, u64::from(policy.format.abi().2))?;
        check_span(
            facts.weight_byte_offset,
            mul(n, weight_row_bytes)?,
            facts.weight_available_bytes,
        )?;
        let native =
            native.ok_or("eligible upstream projection has no native implementation plan")?;
        if native.implementation_fingerprint.is_empty()
            || native.device_architecture < 800
            || native.multiprocessors == 0
        {
            return Err("native implementation and supported device identity are required".into());
        }
        let (padded_k, packed_bytes, fixup_bytes) = match (
            &native.geometry,
            selection.arithmetic.finite_expression(),
        ) {
            (
                UpstreamNativeGeometry::Mmq {
                    padded_inputs,
                    row_tile,
                    column_tile,
                    threads,
                    shared_bytes,
                    packed_guard_blocks,
                    blocks,
                    fixup,
                },
                UpstreamProjectionArithmetic::MmqD4V1 | UpstreamProjectionArithmetic::MmqDs4V1,
            ) => {
                if (facts.local_rows > 32
                    && (*row_tile != 32
                        || *threads != 256
                        || *packed_guard_blocks > 512
                        || (*fixup && *blocks != native.multiprocessors)))
                    || !matches!(row_tile, 8 | 16 | 24 | 32)
                    || *column_tile != 128
                    || *threads == 0
                    || *threads > 1024
                    || *threads % 32 != 0
                    || *packed_guard_blocks > 512
                    || *blocks == 0
                    || *shared_bytes > native.maximum_dynamic_shared_bytes
                    || facts.layout != UpstreamProjectionLayout::Columns
                {
                    return Err("native MMQ geometry violates the declared dense ABI or shared-memory limit".into());
                }
                let pk = u64::from(*padded_inputs);
                let packed_bytes = add(
                    mul(mul(m, pk / 128)?, 144)?,
                    mul(u64::from(*packed_guard_blocks), 144)?,
                )?;
                // The pinned loader copies complete thread batches, even for
                // unused columns in the final row tile. A guard need not be a
                // multiple of eight and can exceed M. Check the actual read
                // endpoint; output write predicates do not protect these reads.
                let tile = u64::from(*row_tile);
                let last_tile = mul((m - 1) / tile, tile)?;
                let copy_bytes = mul(round_up(mul(tile, 36)?, u64::from(*threads))?, 4)?;
                let last_k_block = k
                    .checked_div(128)
                    .and_then(|v| v.checked_sub(1))
                    .ok_or("MMQ requires a complete logical K128 block")?;
                let read_end = add(
                    mul(add(mul(last_k_block, m)?, last_tile)?, 144)?,
                    copy_bytes,
                )?;
                if read_end > packed_bytes {
                    return Err(
                        "native MMQ packed extent does not cover cooperative tail reads".into(),
                    );
                }
                (
                    pk,
                    packed_bytes,
                    if *fixup {
                        mul(
                            mul(
                                mul(u64::from(*blocks), u64::from(*row_tile))?,
                                u64::from(*column_tile),
                            )?,
                            4,
                        )?
                    } else {
                        0
                    },
                )
            }
            (
                UpstreamNativeGeometry::Mmvq {
                    padded_inputs,
                    padded_outputs,
                    columns,
                    channels,
                    warps,
                    rows_per_block,
                },
                UpstreamProjectionArithmetic::MmvqQ8_1V1,
            ) => {
                let expected = match facts.layout {
                    UpstreamProjectionLayout::Columns => (facts.local_rows, 1),
                    UpstreamProjectionLayout::Channels => (1, facts.local_rows),
                };
                if (*columns, *channels) != expected
                    || *warps == 0
                    || *warps > 32
                    || !matches!(rows_per_block, 1 | 2 | 4)
                    || u64::from(*padded_outputs) != round_up(n, u64::from(*rows_per_block))?
                {
                    return Err(
                        "native MMVQ geometry differs from actual columns/channels or tail rows"
                            .into(),
                    );
                }
                if facts.retained_zero_padded_weight_rows < u64::from(*padded_outputs) {
                    return strict(UpstreamStrictReason::RetainedWeightPaddingNotProven);
                }
                check_span(
                    facts.weight_byte_offset,
                    mul(u64::from(*padded_outputs), weight_row_bytes)?,
                    facts.weight_available_bytes,
                )?;
                if mul(u64::from(*padded_outputs), weight_blocks_per_row)? > i32::MAX as u64 {
                    return Err("MMVQ padded weight indexing overflows the native ABI".into());
                }
                let pk = u64::from(*padded_inputs);
                (pk, mul(mul(m, pk / 32)?, 36)?, 0)
            }
            _ => return Err("native geometry and numerical route disagree".into()),
        };
        // MATRIX_ROW_PADDING=512 in the versioned upstream ABI; this is not a
        // tuning threshold. A future native ABI requires a versioned change.
        if padded_k != round_up(k, 512)?
            || mul(m, padded_k)? > i32::MAX as u64
            || mul(mul(m, padded_k)?, 9)? / 8 > i32::MAX as u64
            || mul(m, n)? > i32::MAX as u64
            || mul(n, weight_blocks_per_row)? > i32::MAX as u64
        {
            return Err("upstream native dimension or indexing ABI overflows".into());
        }
        let mut scratch = Vec::new();
        let mut size = 0;
        for (role, bytes) in [
            (
                UpstreamScratchRole::ConvertedInput,
                mul(mul(m, padded_k)?, 4)?,
            ),
            (UpstreamScratchRole::PackedActivations, packed_bytes),
            (UpstreamScratchRole::ProjectionOutput, mul(mul(m, n)?, 4)?),
            (UpstreamScratchRole::StreamKFixup, fixup_bytes),
            (
                UpstreamScratchRole::RowPoisonFlags,
                if selection.arithmetic.is_marker_v2() {
                    mul(m, 4)?
                } else {
                    0
                },
            ),
        ] {
            if bytes == 0 {
                continue;
            }
            let offset = round_up(size, 16)?;
            size = add(offset, bytes)?;
            scratch.push(UpstreamScratchRegion {
                role,
                byte_offset: offset,
                byte_len: bytes,
            });
        }
        Ok(Self {
            prepared_numerics: prepared.clone(),
            facts: facts.clone(),
            route: PreparedUpstreamProjectionRoute::Selected {
                format: policy.format,
                arithmetic: selection.arithmetic,
                pack: selection.arithmetic.semantics(policy.format)?.pack,
                native: native.clone(),
                scratch,
                scratch_bytes: size,
                retained_weight_validation: selection.arithmetic.is_marker_v2().then(|| {
                    UpstreamRetainedWeightValidation {
                        component_id: facts.component_id.clone(),
                        weight_byte_offset: facts.weight_byte_offset,
                        arithmetic: selection.arithmetic,
                        format: policy.format,
                        input_features: k,
                        output_features: n,
                        byte_len: 4,
                        alignment: 4,
                    }
                }),
            },
        })
    }

    pub fn validate_reconstructed(
        &self,
        contract: &super::super::CompositeNumericalArithmetic,
        values: &[crate::vnext::ResolvedValueBinding],
        facts: &UpstreamProjectionWaveFacts,
        native: Option<&UpstreamNativePlanFacts>,
    ) -> Result<(), String> {
        let prepared = PreparedProjectionNumerics::prepare(contract, values)?;
        if self != &Self::prepare(&prepared, facts, native)? {
            return Err("upstream wave plan differs from trusted reconstruction".into());
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Static declaration is insufficient to execute. The implementation must
    /// provide the exact format/route and its device value handling, including
    /// replay. V1 requires domain rejection; V2 requires row/leaf marker and cast
    /// propagation. This declaration API implements neither and is not evidence
    /// of a GPU gate. Neither route permits value-driven strict fallback.
    pub fn require_execution_support(
        &self,
        available: &BTreeSet<(ProjectionBlockFormat, UpstreamProjectionArithmetic)>,
        domain: UpstreamDynamicDomainSupport,
    ) -> Result<(), String> {
        if matches!(self.route, PreparedUpstreamProjectionRoute::G32 { .. }) {
            return Err("G32 needs its separate complete G32 kernel capability; native MarkerV2 support does not qualify it".into());
        }
        if let PreparedUpstreamProjectionRoute::Selected {
            format, arithmetic, ..
        } = self.route
        {
            if !available.contains(&(format, arithmetic)) {
                return Err(
                    "eligible upstream route export unavailable; strict fallback forbidden".into(),
                );
            }
            let required = if arithmetic.is_marker_v2() {
                UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device
            } else {
                UpstreamDynamicDomainSupport::EnforcedBeforeDot
            };
            if domain != required {
                return Err("upstream dynamic numerical domain has no device enforcement".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamDynamicDomainSupport {
    Unavailable,
    /// Provider promises actual device validation (including graph replay),
    /// reporting failure before dot for values outside the selected domain.
    EnforcedBeforeDot,
    /// Actual V2 pack clears/writes row flags, prepared validation owns a leaf
    /// flag, and V2 cast consumes both and canonicalizes exceptional outputs.
    /// Compiled symbols alone do not establish that this protocol is wired.
    CanonicalNanMarkerV2Device,
}

fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "upstream byte extent overflow".into())
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| "upstream byte extent overflow".into())
}
fn round_up(n: u64, a: u64) -> Result<u64, String> {
    if a == 0 {
        return Err("zero alignment".into());
    }
    mul(add(n, a - 1)? / a, a)
}
fn check_span(offset: u64, required: u64, available: u64) -> Result<(), String> {
    add(offset, required)?;
    if required > available {
        Err("upstream view does not own the required byte range".into())
    } else {
        Ok(())
    }
}
