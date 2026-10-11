//! Static projection decisions retained by a plan, never runtime-use counters.
use super::{
    CompositeNumericalArithmetic, DeclaredProjectionArithmetic, ProjectionRole,
    StrictProjectionReason,
};
use crate::vnext::{
    ElementType, PhysicalWeightLayout, PhysicalWeightPadding, ResolvedValueBinding,
    ResolvedValueRole, ResolvedWeightBinding, WeightEncoding, WeightId,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID, DENSE_SWIGLU_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
pub(super) mod tests;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedProjectionRoute {
    Staged {},
    StrictBase { reason: StrictProjectionReason },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedProjectionLeaf {
    component_id: WeightId,
    encoding: WeightEncoding,
    output_features: u64,
    output_offset: u64,
    has_weight_transform: bool,
    route: PreparedProjectionRoute,
}

impl PreparedProjectionLeaf {
    pub fn component_id(&self) -> &WeightId {
        &self.component_id
    }
    pub fn encoding(&self) -> &WeightEncoding {
        &self.encoding
    }
    pub fn output_features(&self) -> u64 {
        self.output_features
    }
    pub fn output_offset(&self) -> u64 {
        self.output_offset
    }
    pub fn has_weight_transform(&self) -> bool {
        self.has_weight_transform
    }
    pub fn route(&self) -> &PreparedProjectionRoute {
        &self.route
    }
    pub fn is_staged(&self) -> bool {
        matches!(self.route, PreparedProjectionRoute::Staged {})
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedProjection {
    role: ProjectionRole,
    weight_input_ordinal: u32,
    input_features: u64,
    output_features: u64,
    leaves: Vec<PreparedProjectionLeaf>,
}

impl PreparedProjection {
    pub fn role(&self) -> ProjectionRole {
        self.role
    }
    pub fn weight_input_ordinal(&self) -> u32 {
        self.weight_input_ordinal
    }
    pub fn input_features(&self) -> u64 {
        self.input_features
    }
    pub fn output_features(&self) -> u64 {
        self.output_features
    }
    pub fn leaves(&self) -> &[PreparedProjectionLeaf] {
        &self.leaves
    }
    pub fn has_staged_leaf(&self) -> bool {
        self.leaves.iter().any(PreparedProjectionLeaf::is_staged)
    }
}

/// Deserializing this value does not establish trust. Plan construction and
/// revalidation reconstruct it from the profile and exact physical bindings.
#[derive(Debug, Clone)]
pub struct PreparedProjectionNumerics {
    data: Arc<PreparedProjectionNumericsData>,
}

// This data has no mutable production access and deliberately is not Clone:
// callers cannot copy validated caches and then mutate their associated data.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename = "PreparedProjectionNumerics", deny_unknown_fields)]
struct PreparedProjectionNumericsData {
    contract: CompositeNumericalArithmetic,
    projections: Vec<PreparedProjection>,
    #[serde(skip)]
    fingerprint: OnceLock<String>,
    #[serde(skip)]
    static_contract_validation: OnceLock<Result<(), String>>,
}

impl Serialize for PreparedProjectionNumerics {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.data.as_ref().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PreparedProjectionNumerics {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Both skipped caches start empty. Deserialization establishes neither
        // a valid static contract nor trust in physical bindings.
        Ok(Self {
            data: Arc::new(PreparedProjectionNumericsData::deserialize(deserializer)?),
        })
    }
}

impl PartialEq for PreparedProjectionNumerics {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            || (self.data.contract == other.data.contract
                && self.data.projections == other.data.projections)
    }
}

impl Eq for PreparedProjectionNumerics {}

impl PreparedProjectionNumerics {
    /// Stable identity of the declaration and exact physical projection leaves.
    /// Cached only on this immutable prepared value; live backing, generation,
    /// lease and per-wave shape authorization remain separate mandatory checks.
    /// The cache is absent from the wire format and does not establish trust in
    /// a deserialized value: callers must still use `validate_bindings`.
    pub fn fingerprint(&self) -> &str {
        self.data.fingerprint.get_or_init(|| {
            let bytes = serde_json::to_vec(self)
                .expect("prepared projection numerics has a canonical JSON representation");
            format!("{:x}", Sha256::digest(bytes))
        })
    }

    /// Memoizes only the immutable declaration check. This never validates
    /// physical bindings, live views, wave ranges, leases, or native facts.
    pub(super) fn validate_static_contract(&self) -> Result<(), String> {
        self.data
            .static_contract_validation
            .get_or_init(|| self.data.contract.validate())
            .clone()
    }

    pub fn contract(&self) -> &CompositeNumericalArithmetic {
        &self.data.contract
    }
    pub fn projections(&self) -> &[PreparedProjection] {
        &self.data.projections
    }
    pub fn projection(&self, role: ProjectionRole) -> Option<&PreparedProjection> {
        self.data
            .projections
            .iter()
            .find(|projection| projection.role == role)
    }
    pub fn staged_leaf_count(&self) -> usize {
        self.data
            .projections
            .iter()
            .flat_map(|p| &p.leaves)
            .filter(|leaf| leaf.is_staged())
            .count()
    }

    pub fn prepare(
        contract: &CompositeNumericalArithmetic,
        values: &[ResolvedValueBinding],
    ) -> Result<Self, String> {
        contract.validate()?;
        let ports: &[(ProjectionRole, u32, usize)] =
            match contract.strict_base.operation_id.as_str() {
                DENSE_SWIGLU_OPERATION_ID => &[
                    (ProjectionRole::SwiGluGateUp, 1, 3),
                    (ProjectionRole::SwiGluDown, 2, 2),
                ],
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID => &[
                    (ProjectionRole::GatedDeltaInput, 2, 2),
                    (ProjectionRole::GatedDeltaOutput, 7, 2),
                ],
                CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID => &[
                    (ProjectionRole::CausalQuery, 2, 2),
                    (ProjectionRole::CausalKey, 3, 2),
                    (ProjectionRole::CausalValue, 4, 2),
                    (ProjectionRole::CausalOutput, 5, 2),
                ],
                _ => return Err("unsupported strict projection base".into()),
            };
        let mut projections = Vec::new();
        for &(role, ordinal, rank) in ports {
            let mut matching = values.iter().filter(|value| {
                value.role() == ResolvedValueRole::Input && value.ordinal() == ordinal
            });
            let value = matching
                .next()
                .ok_or("projection weight binding is missing")?;
            if matching.next().is_some()
                || value.tensor().element_type() != ElementType::F16
                || value.tensor().dimensions().len() != rank
            {
                return Err("projection weight binding differs from its standard port".into());
            }
            let shape = value.tensor().dimensions();
            let input_features = *shape.last().ok_or("projection has no input dimension")?;
            let output_features = extent(&shape[..shape.len() - 1])?;
            let weight = value
                .weight()
                .ok_or("projection binding has no physical weight metadata")?;
            let mut leaves = Vec::new();
            flatten(
                weight,
                weight.physical_layout(),
                shape,
                0,
                false,
                &mut leaves,
            )?;
            leaves.sort_by_key(|leaf| leaf.output_offset);
            let mut cursor = 0_u64;
            for leaf in &mut leaves {
                if leaf.output_offset != cursor {
                    return Err("projection parts overlap or leave a gap".into());
                }
                cursor = cursor
                    .checked_add(leaf.output_features)
                    .ok_or("projection output extent overflows")?;
                let block = match &leaf.encoding {
                    WeightEncoding::BlockQuantized(spec) => Some(spec),
                    _ => None,
                };
                leaf.route = match contract.declared_projection_arithmetic(
                    role,
                    block,
                    input_features,
                    leaf.output_features,
                    leaf.has_weight_transform,
                )? {
                    DeclaredProjectionArithmetic::Staged(_) => PreparedProjectionRoute::Staged {},
                    DeclaredProjectionArithmetic::StrictBase(reason) => {
                        PreparedProjectionRoute::StrictBase { reason }
                    }
                };
            }
            if cursor != output_features {
                return Err("projection parts do not cover the weight output".into());
            }
            projections.push(PreparedProjection {
                role,
                weight_input_ordinal: ordinal,
                input_features,
                output_features,
                leaves,
            });
        }
        Ok(Self {
            data: Arc::new(PreparedProjectionNumericsData {
                contract: contract.clone(),
                projections,
                fingerprint: OnceLock::new(),
                static_contract_validation: OnceLock::from(Ok(())),
            }),
        })
    }

    pub fn validate_bindings(
        &self,
        contract: &CompositeNumericalArithmetic,
        values: &[ResolvedValueBinding],
    ) -> Result<(), String> {
        if self != &Self::prepare(contract, values)? {
            return Err(
                "prepared projection arithmetic differs from declared policy or physical bindings"
                    .into(),
            );
        }
        Ok(())
    }
}

fn extent(shape: &[u64]) -> Result<u64, String> {
    if shape.is_empty() {
        return Err("projection matrix has an empty shape".into());
    }
    shape.iter().try_fold(1_u64, |size, &dim| {
        size.checked_mul(dim)
            .filter(|&size| size > 0)
            .ok_or_else(|| "projection matrix extent is zero or overflows".into())
    })
}

fn flatten(
    weight: &ResolvedWeightBinding,
    layout: &PhysicalWeightLayout,
    shape: &[u64],
    output_offset: u64,
    transformed: bool,
    leaves: &mut Vec<PreparedProjectionLeaf>,
) -> Result<(), String> {
    if let PhysicalWeightLayout::Hadamard { values, .. } = layout {
        if transformed {
            return Err("nested projection transforms are unsupported".into());
        }
        return flatten(weight, values, shape, output_offset, true, leaves);
    }
    if let PhysicalWeightLayout::Composite { parts } = layout {
        for part in parts {
            if part.extents.len() != shape.len()
                || part.logical_offsets.len() != shape.len()
                || part.extents.last() != shape.last()
                || part.logical_offsets.last() != Some(&0)
            {
                return Err("projection part must preserve complete input rows".into());
            }
            let (mut start, mut end) = (0_u64, 0_u64);
            for ((&offset, &dim), &total) in part
                .logical_offsets
                .iter()
                .zip(&part.extents)
                .zip(shape)
                .take(shape.len() - 1)
            {
                let limit = offset
                    .checked_add(dim)
                    .filter(|&limit| dim > 0 && limit <= total)
                    .ok_or("projection part exceeds logical shape")?;
                start = start
                    .checked_mul(total)
                    .and_then(|v| v.checked_add(offset))
                    .ok_or("projection part offset overflows")?;
                end = end
                    .checked_mul(total)
                    .and_then(|v| v.checked_add(limit - 1))
                    .ok_or("projection part end overflows")?;
            }
            if end.checked_sub(start).and_then(|v| v.checked_add(1))
                != Some(extent(&part.extents[..shape.len() - 1])?)
            {
                return Err("projection part rows are not contiguous".into());
            }
            flatten(
                weight,
                &part.layout,
                &part.extents,
                output_offset
                    .checked_add(start)
                    .ok_or("projection part offset overflows")?,
                transformed,
                leaves,
            )?;
        }
        return Ok(());
    }
    let (id, blocked) = match layout {
        PhysicalWeightLayout::Dense { component_id } => (component_id, false),
        PhysicalWeightLayout::Stored { component } => (&component.component_id, false),
        PhysicalWeightLayout::BlockQuantized {
            blocks,
            block_axis,
            block_padding,
        } if *block_axis as usize == shape.len() - 1
            && *block_padding == PhysicalWeightPadding::Exact =>
        {
            (&blocks.component_id, true)
        }
        _ => return Err("projection layout is not a native complete-row matrix".into()),
    };
    let component = weight
        .components()
        .iter()
        .find(|component| component.component_id() == id)
        .ok_or("projection component is absent")?;
    let input_features = *shape.last().ok_or("projection matrix input is absent")?;
    let output_features = extent(&shape[..shape.len() - 1])?;
    let physical_columns = match component.encoding() {
        WeightEncoding::Dense {
            element_type: ElementType::F16,
        } if !blocked => input_features,
        WeightEncoding::BlockQuantized(spec) if blocked => {
            spec.validate().map_err(|error| error.to_string())?;
            let block = u64::from(spec.logical_values_per_block);
            if input_features == 0 || input_features % block != 0 {
                return Err("projection row ends inside a weight block".into());
            }
            input_features / block
        }
        _ => return Err("projection component encoding disagrees with its layout".into()),
    };
    if component.physical_dimensions().last() != Some(&physical_columns)
        || extent(component.physical_dimensions())?
            != output_features
                .checked_mul(physical_columns)
                .ok_or("projection physical extent overflows")?
    {
        return Err("projection physical dimensions differ from logical rows".into());
    }
    leaves.push(PreparedProjectionLeaf {
        component_id: id.clone(),
        encoding: component.encoding().clone(),
        output_features,
        output_offset,
        has_weight_transform: transformed,
        route: PreparedProjectionRoute::StrictBase {
            reason: StrictProjectionReason::UnmodifiedProjection,
        },
    });
    Ok(())
}
