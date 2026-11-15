//! Final AdaLN output conditioning from the unfused MiniMaxH3AdaLayerNormOut
//! path. The cold table producer and request normalization consumer have
//! separate families and lifetimes. No allocator or cache is introduced.
use super::{
    FastH3ConditioningSchedule, FastH3ConditioningTable, FastH3PackedLayout,
    FastH3TransformerPackage, FastH3WeightStorage,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod family;
mod normalization;
mod table;
mod time_projection;
pub use family::{FastH3OutputNormFamily, FastH3OutputTimeFamily};
pub use normalization::FastH3OutputNormConfig;
pub use table::FastH3OutputConditioningTable;
pub use time_projection::FastH3OutputTimeConfig;

#[derive(Clone, Copy, Debug)]
pub struct FastH3OutputConditioningLimits {
    pub maximum_rows: u64,
    pub maximum_time_rows: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}
impl FastH3OutputConditioningLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_rows == 0
            || self.maximum_time_rows == 0
            || self.maximum_metadata_bytes == 0
            || self.maximum_stored_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}

/// Exact original dense parameters selected by either final-conditioning
/// family. The source forwards its archive identity and retained Host owner.
pub struct FastH3OutputConditioningSource<'a> {
    package: &'a FastH3TransformerPackage,
    components: Vec<WeightComponentSpec>,
}
impl<'a> FastH3OutputConditioningSource<'a> {
    pub fn for_time_projection(
        package: &'a FastH3TransformerPackage,
        c: &FastH3OutputTimeConfig,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<Self, VNextError> {
        c.validate(limits)?;
        if c.embedding != package.config().time_embed_dim() as u64
            || c.hidden != package.config().hidden_size() as u64
            || package.config().adaln_rank().is_some()
            || c.components != time_projection::components(package)?
        {
            return Err(invalid("output time source differs from checkpoint"));
        }
        Ok(Self {
            package,
            components: c.components.to_vec(),
        })
    }
    pub fn for_normalization(
        package: &'a FastH3TransformerPackage,
        c: &FastH3OutputNormConfig,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<Self, VNextError> {
        c.validate(limits)?;
        if c.hidden != package.config().hidden_size() as u64
            || c.gamma != dense(package, "norm_out.norm.weight")?
            || c.epsilon
                != CanonicalRational::from_decimal_str(
                    &package.config().final_norm_eps().to_string(),
                )?
        {
            return Err(invalid("final norm source differs from checkpoint"));
        }
        Ok(Self {
            package,
            components: vec![c.gamma.clone()],
        })
    }
}
impl WeightComponentSource for FastH3OutputConditioningSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if !self.components.contains(c) {
            return Err(invalid("undeclared final conditioning component"));
        }
        self.package.dense_archive().component(c)
    }
}
fn dense(
    package: &FastH3TransformerPackage,
    name: &str,
) -> Result<WeightComponentSpec, VNextError> {
    let w = package
        .weight(name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    let FastH3WeightStorage::Dense(c) = &w.storage else {
        return Err(invalid("final conditioning weights must remain dense"));
    };
    if w.compute_element_type != ElementType::Bf16
        || c.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            })
    {
        return Err(invalid(
            "requires original BF16 final conditioning parameters",
        ));
    }
    Ok(c.clone())
}
fn check_component(c: &WeightComponentSpec, shape: &[u64]) -> Result<(), VNextError> {
    if c.dimensions != shape
        || c.role != WeightComponentRole::Values
        || !c.required
        || c.external_names.len() != 1
        || c.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            })
    {
        return Err(invalid(
            "final conditioning parameter shape, role or encoding differs",
        ));
    }
    Ok(())
}
fn check_times(times: &[f32], limits: FastH3OutputConditioningLimits) -> Result<(), VNextError> {
    limits.validate()?;
    if times.is_empty()
        || times.len() > limits.maximum_time_rows
        || times
            .iter()
            .any(|t| !t.is_finite() || !(0.0..=1.0).contains(t))
        || times.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid(
            "requires bounded strictly increasing finite flow times",
        ));
    }
    Ok(())
}
fn same_times(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn check_capacity(
    c: &impl Serialize,
    components: &[WeightComponentSpec],
    limits: FastH3OutputConditioningLimits,
) -> Result<(), VNextError> {
    let n = components.iter().try_fold(0u64, |n, c| {
        n.checked_add(c.physical_bytes()?)
            .ok_or_else(|| invalid("source bytes overflow"))
    })?;
    if n > limits.maximum_stored_bytes
        || serde_json::to_vec(c).map_err(invalid)?.len() > limits.maximum_metadata_bytes
    {
        return Err(invalid("source or metadata capacity exceeded"));
    }
    Ok(())
}
fn schema(prefix: &str, components: &[WeightComponentSpec]) -> Result<WeightSchema, VNextError> {
    let s = WeightSchema {
        format_id: id("weight-format.safetensors.dense")?,
        layout_id: id(&format!("weight-layout.fast_h3.{prefix}.v1"))?,
        version: ContractVersion::new(1, 0),
        components: components.to_vec(),
        tensors: components
            .iter()
            .enumerate()
            .map(|(i, c)| {
                Ok(WeightTensorSpec {
                    id: id(&format!("weight.fast_h3.{prefix}.{i}"))?,
                    dimensions: c.dimensions.clone(),
                    logical_element_type: ElementType::Bf16,
                    physical_layout: PhysicalWeightLayout::Dense {
                        component_id: c.id.clone(),
                    },
                    required: true,
                })
            })
            .collect::<Result<_, VNextError>>()?,
    };
    s.validate(&id(&format!("family.fast_h3.{prefix}"))?)?;
    Ok(s)
}
fn references(
    prefix: &str,
    components: &[WeightComponentSpec],
) -> Result<Vec<WeightReference>, VNextError> {
    components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            Ok(WeightReference {
                weight_id: id(&format!("weight.fast_h3.{prefix}.{i}"))?,
                value_id: id(&format!("value.fast_h3.{prefix}.weight{i}"))?,
                tensor: tensor(c.dimensions.clone(), ElementType::Bf16),
            })
        })
        .collect()
}
fn node(
    prefix: &str,
    name: &str,
    op: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: id(&format!("node.fast_h3.{prefix}.{name}"))?,
        operation_id: id(op)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn resolved(dimensions: Vec<u64>, dtype: ElementType) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(dimensions, dtype, ResolvedTensorLayout::Contiguous)
}
fn managed_copy(
    host: &HostMemoryAccount,
    name: &str,
    shape: Vec<u64>,
    dtype: ElementType,
    bytes: &[u8],
) -> Result<ManagedHostTensor, VNextError> {
    let spec = resolved(shape, dtype)?;
    if spec.minimum_storage_bytes()? != bytes.len() as u64 {
        return Err(invalid("managed copy extent differs"));
    }
    let mut owned = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .map_err(invalid)?;
    owned.as_mut().copy_from_slice(bytes);
    ManagedHostTensor::new(id(name)?, spec, owned.freeze())
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 output conditioning: {e}"),
    }
}
