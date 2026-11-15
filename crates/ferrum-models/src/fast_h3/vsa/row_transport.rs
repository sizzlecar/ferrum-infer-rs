//! Immutable checked VSA layout -> ordinary managed I32 map -> shared tensor
//! transport. The operation's total zero-padding rule is separate from the
//! model proof that every packed row has one inverse tile slot.
use super::*;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3VsaRowTransport {
    Tile,
    Untile,
    /// Expand one row per tile directly into original packed row order.
    ExpandTileValues,
    /// Compact valid tile slots in ascending tile order. This is the prefix
    /// attention key order, which differs from restoring original packed rows.
    ValidTileRows,
    /// All dense-prefix query slots, including padding, in tile order.
    PrefixQueryRows,
    /// All sparse-video query slots after the prefix, including padding.
    VideoQueryRows,
}
#[derive(Clone, Copy, Debug)]
pub struct FastH3VsaRowTransportLimits {
    pub maximum_rows: u64,
    pub maximum_columns: u64,
    pub maximum_batches: u64,
    pub maximum_map_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3VsaRowTransportLimits {
    fn validate(self) -> Result<(), VNextError> {
        if [
            self.maximum_rows,
            self.maximum_columns,
            self.maximum_batches,
            self.maximum_map_bytes,
        ]
        .contains(&0)
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3VsaRowTransportConfig {
    kind: FastH3VsaRowTransport,
    batches: u64,
    columns: u64,
    packed_rows: u64,
    tile_rows: u64,
    map_sha256: String,
    component: WeightComponentSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    query_range: Option<std::ops::Range<u64>>,
}
impl FastH3VsaRowTransportConfig {
    /// Share an immutable row map across tensors with a different channel
    /// width. Only activation geometry changes; physical map identity does not.
    pub fn with_columns(
        &self,
        columns: u64,
        limits: FastH3VsaRowTransportLimits,
    ) -> Result<Self, VNextError> {
        let mut c = self.clone();
        c.columns = columns;
        c.validate(limits)?;
        Ok(c)
    }
    pub fn kind(&self) -> FastH3VsaRowTransport {
        self.kind
    }
    pub fn batches(&self) -> u64 {
        self.batches
    }
    pub fn columns(&self) -> u64 {
        self.columns
    }
    pub fn source_rows(&self) -> u64 {
        match self.kind {
            FastH3VsaRowTransport::Tile => self.packed_rows,
            FastH3VsaRowTransport::Untile
            | FastH3VsaRowTransport::ValidTileRows
            | FastH3VsaRowTransport::PrefixQueryRows
            | FastH3VsaRowTransport::VideoQueryRows => self.tile_rows,
            FastH3VsaRowTransport::ExpandTileValues => {
                self.tile_rows / FAST_H3_VSA_TILE_ROWS as u64
            }
        }
    }
    pub fn output_rows(&self) -> u64 {
        match self.kind {
            FastH3VsaRowTransport::Tile => self.tile_rows,
            FastH3VsaRowTransport::PrefixQueryRows | FastH3VsaRowTransport::VideoQueryRows => self
                .query_range
                .as_ref()
                .map_or(0, |r| r.end.saturating_sub(r.start)),
            _ => self.packed_rows,
        }
    }
    pub fn compile_options(
        &self,
        limits: FastH3VsaRowTransportLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(BTreeMap::from([
            (
                id(FastH3VsaRowTransportFamily::INPUT_ID)?,
                tensor(
                    vec![self.batches, self.source_rows(), self.columns],
                    ElementType::Bf16,
                ),
            ),
            (
                id(FastH3VsaRowTransportFamily::OUTPUT_ID)?,
                tensor(
                    vec![self.batches, self.output_rows(), self.columns],
                    ElementType::Bf16,
                ),
            ),
        ]))
    }
    fn validate(&self, limits: FastH3VsaRowTransportLimits) -> Result<(), VNextError> {
        limits.validate()?;
        match (self.kind, self.query_range.as_ref()) {
            (FastH3VsaRowTransport::PrefixQueryRows, Some(r))
                if r.start == 0
                    && r.end > 0
                    && r.end < self.tile_rows
                    && r.end % FAST_H3_VSA_TILE_ROWS as u64 == 0 => {}
            (FastH3VsaRowTransport::VideoQueryRows, Some(r))
                if r.end == self.tile_rows
                    && r.start < r.end
                    && r.start % FAST_H3_VSA_TILE_ROWS as u64 == 0 => {}
            (FastH3VsaRowTransport::PrefixQueryRows | FastH3VsaRowTransport::VideoQueryRows, _) => {
                return Err(invalid(
                    "query range must be a nonempty tile-aligned prefix or video suffix",
                ))
            }
            (_, None) => {}
            (_, Some(_)) => return Err(invalid("query range is only valid for query slicing")),
        }
        if self.batches == 0
            || self.batches > limits.maximum_batches
            || self.columns == 0
            || self.columns > limits.maximum_columns
            || self.packed_rows == 0
            || self.packed_rows > self.tile_rows
            || self.tile_rows > limits.maximum_rows
            || self.tile_rows > i32::MAX as u64
            || self.tile_rows % FAST_H3_VSA_TILE_ROWS as u64 != 0
            || self.map_sha256.len() != 64
            || !self
                .map_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || self.component != component(self.kind, self.output_rows(), &self.map_sha256)?
            || self.component.physical_bytes()? > limits.maximum_map_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("invalid map identity, row geometry or capacity"));
        }
        RowIndexGatherBf16Geometry::new(
            &ResolvedTensorSpec::new(
                vec![self.batches, self.source_rows(), self.columns],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?,
            &ResolvedTensorSpec::new(
                vec![self.output_rows()],
                ElementType::I32,
                ResolvedTensorLayout::Contiguous,
            )?,
            &ResolvedTensorSpec::new(
                vec![self.batches, self.output_rows(), self.columns],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?,
        )?;
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.vsa_row_map.i32.v1")?,
            version: ContractVersion::new(1, 0),
            components: vec![self.component.clone()],
            tensors: vec![WeightTensorSpec {
                id: id("weight.fast_h3.vsa_rows.map")?,
                dimensions: vec![self.output_rows()],
                logical_element_type: ElementType::I32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: self.component.id.clone(),
                },
                required: true,
            }],
        })
    }
}

pub struct FastH3VsaRowTransportSource {
    component: WeightComponentSpec,
    bytes: ManagedHostBytes,
}
impl FastH3VsaRowTransportSource {
    pub fn component_spec(&self) -> &WeightComponentSpec {
        &self.component
    }
    pub fn prepare(
        layout: &FastH3VsaLayout,
        kind: FastH3VsaRowTransport,
        batches: u64,
        columns: u64,
        host: &HostMemoryAccount,
        limits: FastH3VsaRowTransportLimits,
    ) -> Result<(FastH3VsaRowTransportConfig, Self), VNextError> {
        limits.validate()?;
        let packed_rows = u64::try_from(layout.packed().sequence_rows()).map_err(invalid)?;
        let tile_rows = u64::try_from(layout.padded_rows()).map_err(invalid)?;
        let prefix = (layout.prefix_tiles() as u64)
            .checked_mul(FAST_H3_VSA_TILE_ROWS as u64)
            .ok_or_else(|| invalid("prefix rows overflow"))?;
        let query_range = match kind {
            FastH3VsaRowTransport::PrefixQueryRows => Some(0..prefix),
            FastH3VsaRowTransport::VideoQueryRows => Some(prefix..tile_rows),
            _ => None,
        };
        let output_rows = match &query_range {
            Some(r) => r
                .end
                .checked_sub(r.start)
                .ok_or_else(|| invalid("query range underflows"))?,
            None => {
                if kind == FastH3VsaRowTransport::Tile {
                    tile_rows
                } else {
                    packed_rows
                }
            }
        };
        // Validate full activation and metadata capacity before admitting map bytes.
        let mut c = FastH3VsaRowTransportConfig {
            kind,
            batches,
            columns,
            packed_rows,
            tile_rows,
            map_sha256: "0".repeat(64),
            component: component(kind, output_rows, &"0".repeat(64))?,
            query_range,
        };
        c.validate(limits)?;
        validate_layout(layout)?;
        let bytes = usize::try_from(c.component.physical_bytes()?).map_err(invalid)?;
        let mut map = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, bytes)
            .map_err(invalid)?;
        let mut valid_slots = layout
            .tile_to_packed_rows()
            .iter()
            .enumerate()
            .filter(|(_, packed)| **packed != FAST_H3_VSA_PADDING_ROW)
            .map(|(slot, _)| slot);
        for (row, chunk) in map.as_mut().chunks_exact_mut(4).enumerate() {
            let value = match kind {
                FastH3VsaRowTransport::Tile => {
                    let v = layout.tile_to_packed_rows()[row];
                    if v == FAST_H3_VSA_PADDING_ROW {
                        -1
                    } else {
                        i32::try_from(v).map_err(invalid)?
                    }
                }
                FastH3VsaRowTransport::Untile => {
                    i32::try_from(layout.packed_to_tile_rows()[row]).map_err(invalid)?
                }
                FastH3VsaRowTransport::ExpandTileValues => {
                    i32::try_from(layout.packed_to_tile_rows()[row] / FAST_H3_VSA_TILE_ROWS as u32)
                        .map_err(invalid)?
                }
                FastH3VsaRowTransport::ValidTileRows => i32::try_from(
                    valid_slots
                        .next()
                        .ok_or_else(|| invalid("valid tile row count is incomplete"))?,
                )
                .map_err(invalid)?,
                FastH3VsaRowTransport::PrefixQueryRows | FastH3VsaRowTransport::VideoQueryRows => {
                    i32::try_from(
                        c.query_range
                            .as_ref()
                            .ok_or_else(|| invalid("query range missing"))?
                            .start
                            + row as u64,
                    )
                    .map_err(invalid)?
                }
            };
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        c.map_sha256 = format!("{:x}", Sha256::digest(map.as_ref()));
        c.component = component(kind, output_rows, &c.map_sha256)?;
        c.validate(limits)?;
        Ok((
            c.clone(),
            Self {
                component: c.component,
                bytes: map.freeze(),
            },
        ))
    }
}
impl WeightComponentSource for FastH3VsaRowTransportSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if c != &self.component {
            return Err(invalid("row map source declaration differs"));
        }
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec!["derived-vsa-row-map-i32".into()],
            c.dimensions.clone(),
            ElementType::I32,
            self.bytes.clone(),
        )
    }
}
fn component(
    kind: FastH3VsaRowTransport,
    rows: u64,
    digest: &str,
) -> Result<WeightComponentSpec, VNextError> {
    let label = match kind {
        FastH3VsaRowTransport::Tile => "tile",
        FastH3VsaRowTransport::Untile => "untile",
        FastH3VsaRowTransport::ExpandTileValues => "expand",
        FastH3VsaRowTransport::ValidTileRows => "valid_tile_rows",
        FastH3VsaRowTransport::PrefixQueryRows => "prefix_queries",
        FastH3VsaRowTransport::VideoQueryRows => "video_queries",
    };
    Ok(WeightComponentSpec {
        id: id(&format!("component.fast_h3.vsa_rows.{label}.{digest}"))?,
        role: WeightComponentRole::Values,
        external_names: vec![format!("derived.vsa_rows.{label}.{digest}")],
        dimensions: vec![rows],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::I32,
        },
        required: true,
    })
}
fn validate_layout(layout: &FastH3VsaLayout) -> Result<(), VNextError> {
    let forward = layout.packed_to_tile_rows();
    let inverse = layout.tile_to_packed_rows();
    if inverse.len() != layout.tile_count() * FAST_H3_VSA_TILE_ROWS
        || forward.len() != layout.packed().sequence_rows()
    {
        return Err(invalid("layout map extents differ"));
    }
    for (packed, &slot) in forward.iter().enumerate() {
        if inverse.get(slot as usize).copied() != Some(packed as u32) {
            return Err(invalid("packed row lacks a unique inverse tile slot"));
        }
    }
    for (slot, &packed) in inverse.iter().enumerate() {
        let valid = slot % FAST_H3_VSA_TILE_ROWS
            < usize::from(layout.valid_rows_per_tile()[slot / FAST_H3_VSA_TILE_ROWS]);
        if valid {
            if packed == FAST_H3_VSA_PADDING_ROW
                || forward.get(packed as usize).copied() != Some(slot as u32)
            {
                return Err(invalid("valid tile slot is not an original packed row"));
            }
        } else if packed != FAST_H3_VSA_PADDING_ROW {
            return Err(invalid("padding slot has a source row"));
        }
    }
    Ok(())
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 VSA row map: {e}"),
    }
}

mod family;
pub use family::FastH3VsaRowTransportFamily;
#[cfg(test)]
mod tests;
