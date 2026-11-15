use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};

use super::super::VNextError;
use super::foundation::invalid_operation;
use super::{ElementType, ResolvedWeightBinding, WeightComponentRole, WeightEncoding};

fn is_axis_permutation(axis_order: &[u32], rank: usize) -> bool {
    axis_order.len() == rank
        && axis_order.iter().copied().collect::<BTreeSet<_>>()
            == (0..rank as u32).collect::<BTreeSet<_>>()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionConstraint {
    Exact(u64),
    Symbol(String),
    Range { minimum: u64, maximum: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrideConstraint {
    ExactBytes(u64),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutConstraint {
    Contiguous,
    Strided {
        strides: Vec<StrideConstraint>,
    },
    Blocked {
        block: Vec<u64>,
        axis_order: Vec<u32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliasPolicy {
    NoAlias,
    MayAlias {
        tensor_index: u32,
    },
    MustAlias {
        tensor_index: u32,
    },
    /// Exact, contiguous storage reinterpretation without a write. The plan
    /// keeps every reader alive and forbids mutation of the entire alias class.
    ReadOnlyView {
        tensor_index: u32,
    },
}

/// Meaning of a Stored weight at this operation port. Ordinary activation
/// tensors remain governed by the tensor shape/dtype contract.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum StoredWeightInterpretation {
    #[default]
    Numerical,
    PhysicalComponent {
        role: WeightComponentRole,
        encoding: WeightEncoding,
    },
    /// A numerical operand can also be supplied as one explicitly checked
    /// physical metadata component. Neither branch changes its interpretation.
    NumericalOrPhysicalComponent {
        role: WeightComponentRole,
        encoding: WeightEncoding,
    },
}

impl StoredWeightInterpretation {
    fn is_numerical(&self) -> bool {
        matches!(self, Self::Numerical)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorContract {
    dimensions: Vec<DimensionConstraint>,
    element_types: BTreeSet<ElementType>,
    layouts: Vec<LayoutConstraint>,
    access: TensorAccess,
    alias: AliasPolicy,
    #[serde(skip_serializing_if = "StoredWeightInterpretation::is_numerical")]
    stored_weight_interpretation: StoredWeightInterpretation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorContractWire {
    dimensions: Vec<DimensionConstraint>,
    element_types: BTreeSet<ElementType>,
    layouts: Vec<LayoutConstraint>,
    access: TensorAccess,
    alias: AliasPolicy,
    #[serde(default)]
    stored_weight_interpretation: StoredWeightInterpretation,
}

impl<'de> Deserialize<'de> for TensorContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TensorContractWire::deserialize(deserializer)?;
        let mut contract = Self::new(
            wire.dimensions,
            wire.element_types,
            wire.layouts,
            wire.access,
            wire.alias,
        )
        .map_err(serde::de::Error::custom)?;
        contract.stored_weight_interpretation = wire.stored_weight_interpretation;
        contract
            .validate("tensor_contract")
            .map_err(serde::de::Error::custom)?;
        Ok(contract)
    }
}

impl TensorContract {
    pub fn new(
        dimensions: Vec<DimensionConstraint>,
        element_types: BTreeSet<ElementType>,
        mut layouts: Vec<LayoutConstraint>,
        access: TensorAccess,
        alias: AliasPolicy,
    ) -> Result<Self, VNextError> {
        layouts.sort();
        layouts.dedup();
        let contract = Self {
            dimensions,
            element_types,
            layouts,
            access,
            alias,
            stored_weight_interpretation: StoredWeightInterpretation::Numerical,
        };
        contract.validate("tensor_contract")?;
        Ok(contract)
    }

    pub fn dimensions(&self) -> &[DimensionConstraint] {
        &self.dimensions
    }

    /// Opt in to the exact raw component role and full encoding. No numerical
    /// weight representation can satisfy this policy by sharing its dtype.
    pub fn with_physical_weight_component(
        mut self,
        role: WeightComponentRole,
        encoding: WeightEncoding,
    ) -> Result<Self, VNextError> {
        self.stored_weight_interpretation =
            StoredWeightInterpretation::PhysicalComponent { role, encoding };
        self.validate("tensor_contract")?;
        Ok(self)
    }

    /// Admit ordinary numerical Stored weights as well as the exact declared
    /// raw component. Use this only when both interpretations are meaningful
    /// for the operand (for example a numerical constant or stored scale).
    /// Packed-weight ports should retain the exclusive physical policy.
    pub fn with_numerical_or_physical_weight_component(
        mut self,
        role: WeightComponentRole,
        encoding: WeightEncoding,
    ) -> Result<Self, VNextError> {
        self.stored_weight_interpretation =
            StoredWeightInterpretation::NumericalOrPhysicalComponent { role, encoding };
        self.validate("tensor_contract")?;
        Ok(self)
    }

    pub fn stored_weight_interpretation(&self) -> &StoredWeightInterpretation {
        &self.stored_weight_interpretation
    }

    pub(crate) fn validate_weight_interpretation(
        &self,
        weight: Option<&ResolvedWeightBinding>,
    ) -> Result<(), VNextError> {
        let Some(weight) = weight else {
            return Ok(());
        };
        match (
            &self.stored_weight_interpretation,
            weight.physical_component_view(),
        ) {
            (
                StoredWeightInterpretation::Numerical
                | StoredWeightInterpretation::NumericalOrPhysicalComponent { .. },
                None,
            ) => Ok(()),
            (
                StoredWeightInterpretation::PhysicalComponent { role, encoding }
                | StoredWeightInterpretation::NumericalOrPhysicalComponent { role, encoding },
                Some(component),
            ) if component.role() == *role && component.encoding() == encoding => Ok(()),
            _ => Err(invalid_operation(
                "stored weight interpretation differs from the operation port contract",
            )),
        }
    }

    pub fn element_types(&self) -> &BTreeSet<ElementType> {
        &self.element_types
    }

    pub fn layouts(&self) -> &[LayoutConstraint] {
        &self.layouts
    }

    pub const fn access(&self) -> TensorAccess {
        self.access
    }

    pub fn alias(&self) -> &AliasPolicy {
        &self.alias
    }

    pub fn validate(&self, field: &str) -> Result<(), VNextError> {
        if let StoredWeightInterpretation::PhysicalComponent { encoding, .. }
        | StoredWeightInterpretation::NumericalOrPhysicalComponent { encoding, .. } =
            &self.stored_weight_interpretation
        {
            let physical_type = match encoding {
                WeightEncoding::Dense { element_type } => *element_type,
                WeightEncoding::Quantized(spec) => {
                    spec.validate()?;
                    ElementType::U8
                }
                WeightEncoding::BlockQuantized(spec) => {
                    spec.validate()?;
                    ElementType::U8
                }
                WeightEncoding::DenseAffine { .. } => {
                    return Err(invalid_operation(
                        "physical component ports cannot apply affine decoding",
                    ))
                }
            };
            if self.access != TensorAccess::Read
                || self.alias != AliasPolicy::NoAlias
                || self.layouts != [LayoutConstraint::Contiguous]
                || self.element_types != BTreeSet::from([physical_type])
            {
                return Err(invalid_operation("physical component ports require read-only contiguous storage with the exact physical dtype"));
            }
        }
        if self.element_types.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("{field} has no allowed element type"),
            });
        }
        if self.layouts.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("{field} has no allowed layout"),
            });
        }
        for (index, dimension) in self.dimensions.iter().enumerate() {
            match dimension {
                DimensionConstraint::Exact(0) | DimensionConstraint::Range { minimum: 0, .. } => {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.dimensions[{index}] permits a zero extent"),
                    });
                }
                DimensionConstraint::Range { minimum, maximum } if minimum > maximum => {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.dimensions[{index}] has an inverted range"),
                    });
                }
                DimensionConstraint::Symbol(symbol) if symbol.trim().is_empty() => {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.dimensions[{index}] has an empty symbol"),
                    });
                }
                _ => {}
            }
        }
        for (index, layout) in self.layouts.iter().enumerate() {
            match layout {
                LayoutConstraint::Strided { strides }
                    if strides.len() != self.dimensions.len()
                        || strides.iter().any(|stride| match stride {
                            StrideConstraint::ExactBytes(bytes) => *bytes == 0,
                            StrideConstraint::Symbol(symbol) => symbol.trim().is_empty(),
                        }) =>
                {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.layouts[{index}] has invalid strides"),
                    });
                }
                LayoutConstraint::Blocked { block, axis_order }
                    if block.len() != self.dimensions.len()
                        || block.iter().any(|extent| *extent == 0)
                        || !is_axis_permutation(axis_order, self.dimensions.len()) =>
                {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.layouts[{index}] has an invalid block"),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockedTensorPadding {
    Exact,
    ZeroFill { physical_dimensions: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedTensorLayout {
    Contiguous,
    Strided {
        byte_strides: Vec<u64>,
    },
    Blocked {
        block: Vec<u64>,
        axis_order: Vec<u32>,
        padding: BlockedTensorPadding,
    },
}

/// Concrete tensor shape selected by planning and consumed unchanged by an
/// operation provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedTensorSpec {
    dimensions: Vec<u64>,
    element_type: ElementType,
    layout: ResolvedTensorLayout,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedTensorSpecWire {
    dimensions: Vec<u64>,
    element_type: ElementType,
    layout: ResolvedTensorLayout,
}

impl<'de> Deserialize<'de> for ResolvedTensorSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedTensorSpecWire::deserialize(deserializer)?;
        Self::new(wire.dimensions, wire.element_type, wire.layout).map_err(serde::de::Error::custom)
    }
}

impl ResolvedTensorSpec {
    pub fn new(
        dimensions: Vec<u64>,
        element_type: ElementType,
        layout: ResolvedTensorLayout,
    ) -> Result<Self, VNextError> {
        if dimensions.iter().any(|extent| *extent == 0) {
            return Err(invalid_operation(
                "resolved tensor dimensions must be non-zero",
            ));
        }
        match &layout {
            ResolvedTensorLayout::Strided { byte_strides }
                if byte_strides.len() != dimensions.len()
                    || byte_strides.iter().any(|stride| *stride == 0) =>
            {
                return Err(invalid_operation(
                    "resolved tensor byte strides must match rank and be non-zero",
                ));
            }
            ResolvedTensorLayout::Blocked {
                block,
                axis_order,
                padding,
            } => {
                if block.len() != dimensions.len()
                    || block.iter().any(|extent| *extent == 0)
                    || !is_axis_permutation(axis_order, dimensions.len())
                {
                    return Err(invalid_operation(
                        "resolved tensor block and axis order must form a non-zero ranked layout",
                    ));
                }
                match padding {
                    BlockedTensorPadding::Exact => {
                        if dimensions
                            .iter()
                            .zip(block)
                            .any(|(extent, block)| extent % block != 0)
                        {
                            return Err(invalid_operation(
                                "exact blocked tensors require every logical extent to be block-divisible",
                            ));
                        }
                    }
                    BlockedTensorPadding::ZeroFill {
                        physical_dimensions,
                    } => {
                        if physical_dimensions.len() != dimensions.len() {
                            return Err(invalid_operation(
                                "zero-filled blocked tensor padding must match tensor rank",
                            ));
                        }
                        let mut has_padding = false;
                        let mut padded_logical = Vec::with_capacity(dimensions.len());
                        for (logical, block) in dimensions.iter().zip(block) {
                            let expected = logical
                                .checked_add(block - 1)
                                .map(|extent| extent / block * block)
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "zero-filled blocked tensor padding overflows u64",
                                    )
                                })?;
                            padded_logical.push(expected);
                            has_padding |= expected != *logical;
                        }
                        let expected_physical = axis_order
                            .iter()
                            .map(|axis| padded_logical[*axis as usize])
                            .collect::<Vec<_>>();
                        if *physical_dimensions != expected_physical {
                            return Err(invalid_operation(
                                "zero-filled blocked tensor physical shape is not the minimal block-aligned axis permutation",
                            ));
                        }
                        if !has_padding {
                            return Err(invalid_operation(
                                "zero-filled blocked tensor layout must contain actual padding; use Exact otherwise",
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
        dimensions
            .iter()
            .try_fold(element_type.size_bytes(), |bytes, extent| {
                bytes.checked_mul(*extent)
            })
            .ok_or_else(|| invalid_operation("resolved tensor byte size overflows u64"))?;
        Ok(Self {
            dimensions,
            element_type,
            layout,
        })
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn layout(&self) -> &ResolvedTensorLayout {
        &self.layout
    }

    pub fn minimum_storage_bytes(&self) -> Result<u64, VNextError> {
        match &self.layout {
            ResolvedTensorLayout::Contiguous => self
                .dimensions
                .iter()
                .try_fold(self.element_type.size_bytes(), |bytes, extent| {
                    bytes.checked_mul(*extent)
                })
                .ok_or_else(|| invalid_operation("resolved tensor byte size overflows u64")),
            ResolvedTensorLayout::Blocked { padding, .. } => {
                let storage_dimensions = match padding {
                    BlockedTensorPadding::Exact => &self.dimensions,
                    BlockedTensorPadding::ZeroFill {
                        physical_dimensions,
                    } => physical_dimensions,
                };
                storage_dimensions
                    .iter()
                    .try_fold(self.element_type.size_bytes(), |bytes, extent| {
                        bytes.checked_mul(*extent)
                    })
                    .ok_or_else(|| {
                        invalid_operation("resolved blocked tensor byte size overflows u64")
                    })
            }
            ResolvedTensorLayout::Strided { byte_strides } => self
                .dimensions
                .iter()
                .zip(byte_strides)
                .try_fold(self.element_type.size_bytes(), |span, (extent, stride)| {
                    extent
                        .checked_sub(1)
                        .and_then(|steps| steps.checked_mul(*stride))
                        .and_then(|bytes| span.checked_add(bytes))
                })
                .ok_or_else(|| invalid_operation("resolved strided tensor span overflows u64")),
        }
    }
}
