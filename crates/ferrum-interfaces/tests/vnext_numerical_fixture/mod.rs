#![allow(dead_code, unused_imports)]

use ferrum_interfaces::vnext::*;

fn id<T: TryFrom<String, Error = VNextError>>(value: impl Into<String>) -> T {
    T::try_from(value.into()).unwrap()
}

/// Existing F32 contract fixtures explicitly choose their one declared ABI.
/// Numerical-policy tests exercise Auto/Require separately.
pub(crate) trait PrepareFixture: ModelFamilyRegistration {
    fn prepare_fixture(&self, raw: &serde_json::Value) -> Result<PreparedModelFamily, VNextError> {
        self.prepare_with_profile(raw, &NumericalProfileId::new("fixture.f32").unwrap())
    }
}
impl<T: ModelFamilyRegistration + ?Sized> PrepareFixture for T {}

pub(crate) fn fixture_f32_profiles(
    family: &ModelFamilyId,
    outputs: &[&str],
    operations: &[&str],
    states: Vec<StateSpec>,
) -> Result<FamilyNumericalProfiles, VNextError> {
    let profile_id = NumericalProfileId::new("fixture.f32").unwrap();
    FamilyNumericalProfiles::new(
        family,
        ContractVersion::new(1, 0),
        vec![NumericalExecutionProfile {
            id: profile_id.clone(),
            family_id: family.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(*outputs.last().unwrap()),
            boundaries: outputs
                .iter()
                .map(|output| (id(*output), ElementType::F32))
                .collect(),
            states,
            operations: operations
                .iter()
                .map(|operation| NumericalOperationContract {
                    operation_id: id(*operation),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                })
                .collect(),
        }],
        vec![profile_id],
    )
}

pub(crate) fn fixture_byte_state(width: u64) -> StateSpec {
    StateSpec {
        id: id("state.cache"),
        value_id: id("value.state"),
        tensor: ProgramTensorSpec {
            dimensions: vec![width],
            element_type: ElementType::U8,
            layout: ResolvedTensorLayout::Contiguous,
        },
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::FixedPerScope,
        initialization: StateInitialization::Zero,
        checkpoint: StateCheckpointCapability::Unsupported,
    }
}
