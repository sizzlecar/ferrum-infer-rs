//! Generic token-scaled causal inputs. These are immutable accounted Host
//! owners, not an additional sequence registry or device resource ledger.
use super::*;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalTokenInputStorage {
    /// Every admitted token, including decode continuation, has actual bytes.
    Exact,
    /// The wave materializes a complete zero-initialized current-token buffer
    /// and copies only the provided nonoverlapping source segments into it.
    ZeroOutsideSegments,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalTokenInputSpec {
    pub value_id: ProgramValueId,
    pub element_type: ElementType,
    /// Fixed row dimensions after the leading token axis; empty means scalar.
    pub row_dimensions: Vec<u64>,
    pub storage: CausalTokenInputStorage,
}
impl CausalTokenInputSpec {
    pub(super) fn row_bytes(&self) -> Result<usize> {
        let bytes = self
            .row_dimensions
            .iter()
            .try_fold(self.element_type.size_bytes(), |n, &d| {
                (d > 0).then(|| n.checked_mul(d)).flatten()
            })
            .ok_or_else(|| {
                FerrumError::config("causal input row dimensions overflow or contain zero")
            })?;
        usize::try_from(bytes)
            .ok()
            .filter(|&n| n > 0 && n <= isize::MAX as usize)
            .ok_or_else(|| FerrumError::config("causal input row exceeds address space"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalInputSchema {
    specs: Vec<CausalTokenInputSpec>,
}
impl CausalInputSchema {
    pub fn new(mut specs: Vec<CausalTokenInputSpec>) -> Result<Self> {
        if specs.is_empty() {
            return Err(FerrumError::config("causal input schema must be nonempty"));
        }
        specs.sort_by(|a, b| a.value_id.cmp(&b.value_id));
        if specs.windows(2).any(|w| w[0].value_id == w[1].value_id) {
            return Err(FerrumError::config(
                "causal input schema contains duplicate value ids",
            ));
        }
        for spec in &specs {
            if !matches!(
                spec.element_type,
                ElementType::F16
                    | ElementType::Bf16
                    | ElementType::F32
                    | ElementType::U8
                    | ElementType::U32
            ) {
                return Err(FerrumError::unsupported(
                    "causal token inputs require scalar floating/U8/U32 elements",
                ));
            }
            spec.row_bytes()?;
        }
        Ok(Self { specs })
    }
    pub fn specs(&self) -> &[CausalTokenInputSpec] {
        &self.specs
    }
    pub(super) fn zero_demands(&self, tokens: usize) -> Result<Vec<usize>> {
        if tokens == 0 {
            return Err(FerrumError::config("causal input wave must contain tokens"));
        }
        self.specs
            .iter()
            .filter(|s| s.storage == CausalTokenInputStorage::ZeroOutsideSegments)
            .map(|s| {
                tokens
                    .checked_mul(s.row_bytes()?)
                    .filter(|&n| n <= isize::MAX as usize)
                    .ok_or_else(|| FerrumError::config("causal wave input extent overflows"))
            })
            .collect()
    }
    pub(super) fn additional_bytes(
        &self,
        host: &HostMemoryAccount,
        tokens: usize,
    ) -> Result<usize> {
        self.zero_demands(tokens)?
            .into_iter()
            .try_fold(0usize, |sum, bytes| {
                sum.checked_add(
                    host.backing_bytes(bytes)
                        .map_err(|e| FerrumError::resource_exhausted(e.to_string()))?,
                )
                .ok_or_else(|| FerrumError::config("causal wave Input backing overflows"))
            })
    }
}

#[derive(Debug, Clone)]
pub struct CausalInputSegment {
    start: usize,
    rows: usize,
    bytes: ManagedHostBytes,
}
impl CausalInputSegment {
    pub fn new(start_token: usize, rows: usize, bytes: ManagedHostBytes) -> Result<Self> {
        if rows == 0 || bytes.is_empty() || start_token.checked_add(rows).is_none() {
            return Err(FerrumError::request_validation(
                "causal input segment must be nonempty and nonoverflowing",
            ));
        }
        Ok(Self {
            start: start_token,
            rows,
            bytes,
        })
    }
}

#[derive(Debug)]
pub struct CausalTokenInput {
    value_id: ProgramValueId,
    segments: Vec<CausalInputSegment>,
}
impl CausalTokenInput {
    pub fn new(value_id: ProgramValueId, segments: Vec<CausalInputSegment>) -> Self {
        Self { value_id, segments }
    }
}

/// Retained by the original admission/sequence until it actually closes. Clone
/// an Arc of this value for retries; constructing a different owner with equal
/// token IDs is not permission to replace an admitted request's conditioning.
#[derive(Debug)]
pub struct CausalRequestInputs {
    schema: CausalInputSchema,
    maximum_tokens: usize,
    values: Vec<CausalTokenInput>,
}
impl CausalRequestInputs {
    pub fn new(
        host: &HostMemoryAccount,
        schema: &CausalInputSchema,
        maximum_tokens: usize,
        mut values: Vec<CausalTokenInput>,
    ) -> Result<Self> {
        if maximum_tokens == 0 {
            return Err(FerrumError::request_validation(
                "causal inputs require positive maximum tokens",
            ));
        }
        values.sort_by(|a, b| a.value_id.cmp(&b.value_id));
        if values.len() != schema.specs.len()
            || values
                .iter()
                .zip(&schema.specs)
                .any(|(v, s)| v.value_id != s.value_id)
        {
            return Err(FerrumError::request_validation(
                "causal request inputs differ from the exact schema",
            ));
        }
        for (value, spec) in values.iter_mut().zip(&schema.specs) {
            value.segments.sort_by_key(|s| s.start);
            let row = spec.row_bytes()?;
            let mut end = 0;
            for segment in &value.segments {
                let next = segment.start.checked_add(segment.rows).ok_or_else(|| {
                    FerrumError::request_validation("causal input segment overflows")
                })?;
                if segment.start < end
                    || next > maximum_tokens
                    || segment.rows.checked_mul(row) != Some(segment.bytes.len())
                    || segment.bytes.class() != HostMemoryClass::InputPreprocess
                    || !host.owns(&segment.bytes)
                    || (spec.storage == CausalTokenInputStorage::Exact && segment.start != end)
                {
                    return Err(FerrumError::request_validation("causal input segment overlaps, exceeds coverage, has foreign Host/type extent, or lacks exact coverage"));
                }
                end = next;
            }
            if spec.storage == CausalTokenInputStorage::Exact && end != maximum_tokens {
                return Err(FerrumError::request_validation(
                    "Exact causal tables must cover the full admitted sequence including decode",
                ));
            }
        }
        Ok(Self {
            schema: schema.clone(),
            maximum_tokens,
            values,
        })
    }
    pub fn schema(&self) -> &CausalInputSchema {
        &self.schema
    }
    pub fn maximum_tokens(&self) -> usize {
        self.maximum_tokens
    }

    /// Select actual immutable conditioning rows for a native wave. Exact
    /// contiguous rows keep their original owner; sparse or segmented rows
    /// allocate one accounted Input buffer. A draft may select the main
    /// sequence's rotary position independently of its own KV append offset.
    pub fn materialize_rows(
        &self,
        host: &HostMemoryAccount,
        value_id: &ProgramValueId,
        range: Range<usize>,
    ) -> Result<(ElementType, ManagedHostBytes)> {
        self.validate_host(host)?;
        if range.start >= range.end || range.end > self.maximum_tokens {
            return Err(FerrumError::request_validation(
                "causal row selection exceeds the original input extent",
            ));
        }
        let (value, spec) = self
            .values
            .iter()
            .zip(&self.schema.specs)
            .find(|(value, _)| &value.value_id == value_id)
            .ok_or_else(|| FerrumError::request_validation("unknown causal input value"))?;
        let row = spec.row_bytes()?;
        if let Some(segment) = value.segments.iter().find(|segment| {
            segment.start <= range.start && segment.start + segment.rows >= range.end
        }) {
            return Ok((
                spec.element_type,
                segment
                    .bytes
                    .slice((range.start - segment.start) * row..(range.end - segment.start) * row)
                    .map_err(|error| FerrumError::backend(error.to_string()))?,
            ));
        }
        let length = range
            .len()
            .checked_mul(row)
            .ok_or_else(|| FerrumError::request_validation("causal row selection overflows"))?;
        let mut buffer = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, length)
            .map_err(|error| FerrumError::resource_exhausted(error.to_string()))?;
        for segment in &value.segments {
            let low = range.start.max(segment.start);
            let high = range.end.min(segment.start + segment.rows);
            if low < high {
                buffer.as_mut()[(low - range.start) * row..(high - range.start) * row]
                    .copy_from_slice(
                        &segment.bytes.bytes()
                            [(low - segment.start) * row..(high - segment.start) * row],
                    );
            }
        }
        Ok((spec.element_type, buffer.freeze()))
    }

    /// Bind Core's accepted-prefix checks to the original immutable sources.
    /// Only row metadata is copied; visual features and position tables keep
    /// the existing accounted owners, including their sparse zero policy.
    pub(super) fn sequence_conditioning(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<SequenceConditioningInputs> {
        self.validate_host(host)?;
        let rows = u64::try_from(self.maximum_tokens)
            .map_err(|_| FerrumError::request_validation("causal input extent exceeds u64"))?;
        let inputs = self
            .values
            .iter()
            .zip(&self.schema.specs)
            .map(|(value, spec)| {
                let mut dimensions = Vec::with_capacity(spec.row_dimensions.len() + 1);
                dimensions.push(rows);
                dimensions.extend_from_slice(&spec.row_dimensions);
                let tensor = ResolvedTensorSpec::new(
                    dimensions,
                    spec.element_type,
                    ResolvedTensorLayout::Contiguous,
                )?;
                let coverage = match spec.storage {
                    CausalTokenInputStorage::Exact => SequenceConditioningCoverage::Exact,
                    CausalTokenInputStorage::ZeroOutsideSegments => {
                        SequenceConditioningCoverage::ZeroOutsideSegments
                    }
                };
                let segments = value
                    .segments
                    .iter()
                    .map(|segment| {
                        SequenceConditioningSegment::new(segment.start, segment.bytes.clone())
                    })
                    .collect::<std::result::Result<Vec<_>, VNextError>>()?;
                SequenceConditioningInput::new(
                    host,
                    value.value_id.clone(),
                    tensor,
                    coverage,
                    segments,
                )
            })
            .collect::<std::result::Result<Vec<_>, VNextError>>()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        SequenceConditioningInputs::new(inputs)
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    pub(super) fn validate_host(&self, host: &HostMemoryAccount) -> Result<()> {
        if self
            .values
            .iter()
            .flat_map(|v| &v.segments)
            .any(|s| !host.owns(&s.bytes))
        {
            return Err(FerrumError::request_validation(
                "causal request inputs belong to a different Host parent",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare(
        &self,
        range: Range<usize>,
        allocate: &mut impl FnMut() -> Result<ManagedHostBuffer>,
    ) -> Result<Vec<PreparedInput>> {
        if range.start >= range.end || range.end > self.maximum_tokens {
            return Err(FerrumError::request_validation(
                "causal input wave exceeds admitted sequence",
            ));
        }
        self.values
            .iter()
            .zip(&self.schema.specs)
            .map(|(value, spec)| {
                let row = spec.row_bytes()?;
                let mut pieces = Vec::new();
                let mut zero = if spec.storage == CausalTokenInputStorage::ZeroOutsideSegments {
                    Some(allocate()?)
                } else {
                    None
                };
                if let Some(buffer) = &zero {
                    if buffer.len()
                        != range
                            .len()
                            .checked_mul(row)
                            .ok_or_else(|| FerrumError::config("causal input wave overflows"))?
                    {
                        return Err(FerrumError::internal(
                            "causal input reservation differs from its exact current span",
                        ));
                    }
                }
                for segment in &value.segments {
                    let start = range.start.max(segment.start);
                    let end = range.end.min(segment.start + segment.rows);
                    if start >= end {
                        continue;
                    }
                    let source = (start - segment.start) * row..(end - segment.start) * row;
                    if let Some(buffer) = &mut zero {
                        buffer.as_mut()[(start - range.start) * row..(end - range.start) * row]
                            .copy_from_slice(&segment.bytes.bytes()[source]);
                    } else {
                        pieces.push(InputPiece {
                            start,
                            rows: end - start,
                            bytes: segment
                                .bytes
                                .slice(source)
                                .map_err(|e| FerrumError::backend(e.to_string()))?,
                        });
                    }
                }
                if let Some(buffer) = zero {
                    pieces.push(InputPiece {
                        start: range.start,
                        rows: range.len(),
                        bytes: buffer.freeze(),
                    });
                }
                Ok(PreparedInput {
                    value_id: value.value_id.clone(),
                    row_bytes: row,
                    element_type: spec.element_type,
                    pieces,
                })
            })
            .collect()
    }
}

pub(super) struct InputPiece {
    start: usize,
    rows: usize,
    bytes: ManagedHostBytes,
}
pub(super) struct PreparedInput {
    value_id: ProgramValueId,
    row_bytes: usize,
    element_type: ElementType,
    pieces: Vec<InputPiece>,
}

#[derive(Debug, Clone)]
pub(super) struct InputPort {
    node: NodeId,
    ordinal: u32,
}
#[derive(Debug, Clone)]
pub(super) struct ResolvedInputs {
    pub(super) schema: CausalInputSchema,
    ports: BTreeMap<ProgramValueId, Vec<InputPort>>,
}
impl ResolvedInputs {
    pub(super) fn resolve(
        executable: &impl ExecutablePlanView,
        schema: &CausalInputSchema,
    ) -> Result<Self> {
        let mut ports = BTreeMap::new();
        for spec in &schema.specs {
            let mut found = Vec::new();
            for node in executable.execution_plan().payload().nodes() {
                for value in node.values().iter().filter(|v| {
                    v.role() == ResolvedValueRole::Input && v.value_id() == &spec.value_id
                }) {
                    let dims = value.tensor().dimensions();
                    let projection = node
                        .work()
                        .token_projection(ResolvedValueRole::Input, value.ordinal());
                    if dims.is_empty()
                        || dims[0] == 0
                        || dims[1..] != spec.row_dimensions
                        || value.tensor().element_type() != spec.element_type
                        || !matches!(value.tensor().layout(), ResolvedTensorLayout::Contiguous)
                        || value.access() != TensorAccess::Read
                        || value.storage().components().len() != 1
                        || value.usage() != BufferUsage::Activations
                        || projection.is_none_or(|p| {
                            p.axis() != 0
                                || p.rank() as usize != dims.len()
                                || p.canonical_extent() != dims[0]
                                || p.value_id() != &spec.value_id
                        })
                    {
                        return Err(FerrumError::model(
                            "compiled causal input differs from its token-scaled schema",
                        ));
                    }
                    found.push(InputPort {
                        node: node.id().clone(),
                        ordinal: value.ordinal(),
                    });
                }
            }
            if found.is_empty() {
                return Err(FerrumError::model(
                    "causal input has no actual compiled consumer",
                ));
            }
            ports.insert(spec.value_id.clone(), found);
        }
        Ok(Self {
            schema: schema.clone(),
            ports,
        })
    }
    pub(super) fn uploads(
        &self,
        row: &[PreparedInput],
        range: Range<u64>,
        participant: u32,
    ) -> Result<Vec<SubmissionWaveInputUpload>> {
        let start = usize::try_from(range.start)
            .map_err(|_| FerrumError::backend("causal input start exceeds host"))?;
        let end = usize::try_from(range.end)
            .map_err(|_| FerrumError::backend("causal input end exceeds host"))?;
        let mut uploads = Vec::new();
        for input in row {
            for piece in &input.pieces {
                let low = start.max(piece.start);
                let high = end.min(piece.start + piece.rows);
                if low >= high {
                    continue;
                }
                let bytes = piece
                    .bytes
                    .slice(
                        (low - piece.start) * input.row_bytes
                            ..(high - piece.start) * input.row_bytes,
                    )
                    .map_err(|e| FerrumError::backend(e.to_string()))?;
                let offset = (low as u64)
                    .checked_mul(input.row_bytes as u64)
                    .ok_or_else(|| FerrumError::backend("causal upload offset overflows"))?;
                let elements = u64::try_from(bytes.len())
                    .map_err(|_| FerrumError::backend("causal upload size exceeds u64"))?
                    / input.element_type.size_bytes();
                let layout = HostTransferLayout::new(input.element_type, elements)
                    .map_err(|e| FerrumError::backend(e.to_string()))?;
                for port in self
                    .ports
                    .get(&input.value_id)
                    .ok_or_else(|| FerrumError::internal("causal input lost its compiled port"))?
                {
                    uploads.push(
                        SubmissionWaveInputUpload::managed_range(
                            port.node.clone(),
                            participant,
                            port.ordinal,
                            offset,
                            layout,
                            bytes.clone(),
                        )
                        .map_err(|e| FerrumError::backend(e.to_string()))?,
                    );
                }
            }
        }
        Ok(uploads)
    }
}

#[cfg(test)]
mod tests;
