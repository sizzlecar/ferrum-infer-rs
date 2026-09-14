//! Shared construction helpers for language-family numerical declarations.
//! These operate on typed dimensions and roles, without constructing a graph.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    ContractVersion, ElementType, FamilyNumericalProfiles, ModelFamilyId,
    NumericalExecutionProfile, NumericalOperationContract, NumericalProfileId, OperationId,
    ProgramTensorSpec, ProgramValueId, ResolvedTensorLayout, StateCapacityDemand,
    StateCheckpointCapability, StateId, StateInitialization, StateLifetime, StateSpec, VNextError,
};

pub(super) fn kv_state(
    layer: u64,
    heads: u64,
    head_dim: u64,
    maximum_tokens: u64,
) -> Result<StateSpec, VNextError> {
    let tensor = ProgramTensorSpec {
        dimensions: vec![2, heads, head_dim],
        element_type: ElementType::F16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let bytes_per_token = tensor.byte_len()?;
    Ok(StateSpec {
        id: StateId::new(format!("state.layer.{layer}.kv"))?,
        value_id: ProgramValueId::new(format!("value.state.layer.{layer}.kv"))?,
        tensor,
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::TokenScaled {
            bytes_per_token,
            maximum_tokens,
        },
        initialization: StateInitialization::None,
        checkpoint: StateCheckpointCapability::Unsupported,
    })
}

/// A family with one existing F16 boundary ABI still explicitly declares all
/// operations and intermediate values. Fused-operation versions preserve their
/// internal conversions (including BF16 expert compute in GPT-OSS).
pub(super) struct F16LanguageProfile {
    profile: NumericalExecutionProfile,
}

impl F16LanguageProfile {
    pub(super) fn new(family: &ModelFamilyId, id: &str) -> Result<Self, VNextError> {
        let primary_activation = ProgramValueId::new("value.hidden.embedding")?;
        Ok(Self {
            profile: NumericalExecutionProfile {
                id: NumericalProfileId::new(id).map_err(|reason| {
                    VNextError::InvalidModelConfig {
                        family_id: family.to_string(),
                        field: "numerical_profile.id".into(),
                        reason,
                    }
                })?,
                version: ContractVersion::new(1, 0),
                family_id: family.clone(),
                boundaries: BTreeMap::from([
                    (primary_activation.clone(), ElementType::F16),
                    (
                        ProgramValueId::new("value.output.final_hidden")?,
                        ElementType::F16,
                    ),
                    (
                        ProgramValueId::new("value.output.logits")?,
                        ElementType::F16,
                    ),
                    (
                        ProgramValueId::new("value.output.greedy_token")?,
                        ElementType::U32,
                    ),
                ]),
                primary_activation,
                states: Vec::new(),
                operations: Vec::new(),
            },
        })
    }

    pub(super) fn boundary(&mut self, value: impl Into<String>) -> Result<(), VNextError> {
        self.profile
            .boundaries
            .insert(ProgramValueId::new(value)?, ElementType::F16);
        Ok(())
    }

    pub(super) fn layer(
        &mut self,
        index: u64,
        roles: &[&str],
        state: StateSpec,
    ) -> Result<(), VNextError> {
        for role in roles {
            self.boundary(format!("value.layer.{index}.{role}"))?;
        }
        self.profile.states.push(state);
        Ok(())
    }

    pub(super) fn operation(
        &mut self,
        id: &str,
        major: u16,
        multiply: bool,
        accumulate: bool,
    ) -> Result<(), VNextError> {
        self.profile.operations.push(NumericalOperationContract {
            operation_id: OperationId::new(id)?,
            version: ContractVersion::new(major, 0),
            multiplication_type: multiply.then_some(ElementType::F32),
            accumulation_type: accumulate.then_some(ElementType::F32),
        });
        Ok(())
    }

    pub(super) fn finish(self) -> Result<FamilyNumericalProfiles, VNextError> {
        let family = self.profile.family_id.clone();
        let preferred = self.profile.id.clone();
        FamilyNumericalProfiles::new(
            &family,
            ContractVersion::new(1, 0),
            vec![self.profile],
            vec![preferred],
        )
    }
}
