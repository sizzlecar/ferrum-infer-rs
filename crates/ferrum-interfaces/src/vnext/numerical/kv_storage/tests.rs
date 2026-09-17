use super::*;
use crate::vnext::*;

fn state(name: &str, shape: Vec<u64>, dtype: ElementType) -> StateSpec {
    let tensor = ProgramTensorSpec {
        dimensions: shape,
        element_type: dtype,
        layout: ResolvedTensorLayout::Contiguous,
    };
    StateSpec {
        id: StateId::new(format!("state.{name}")).unwrap(),
        value_id: ProgramValueId::new(format!("value.{name}")).unwrap(),
        capacity_demand: StateCapacityDemand::TokenScaled {
            bytes_per_token: tensor.byte_len().unwrap(),
            maximum_tokens: 4096,
        },
        tensor,
        lifetime: StateLifetime::Sequence,
        initialization: StateInitialization::None,
        checkpoint: StateCheckpointCapability::CompletedBoundary(StateCheckpointContract::new(
            StateCheckpointContents::PrefixPositions,
            CheckpointInputDependency::ExactTokenPrefix,
        )),
    }
}

fn profile(quantized: bool) -> NumericalExecutionProfile {
    let mut states = vec![state(
        "payload",
        vec![2, 4, 128],
        if quantized {
            ElementType::I8
        } else {
            ElementType::F16
        },
    )];
    let storage = if quantized {
        states.push(state("scales", vec![2, 4], ElementType::F32));
        KvStateStorage::Int8PerTokenHeadF32ScaleV1 {
            payload_state: states[0].id.clone(),
            scale_state: states[1].id.clone(),
        }
    } else {
        KvStateStorage::F16 {
            state: states[0].id.clone(),
        }
    };
    NumericalExecutionProfile {
        id: NumericalProfileId::new(if quantized {
            "fixture.int8"
        } else {
            "fixture.f16"
        })
        .unwrap(),
        family_id: ModelFamilyId::new("family.fixture.kv").unwrap(),
        version: ContractVersion::new(1, 0),
        primary_activation: ProgramValueId::new("value.output").unwrap(),
        boundaries: BTreeMap::from([(
            ProgramValueId::new("value.output").unwrap(),
            ElementType::F16,
        )]),
        states,
        kv_storage: vec![storage],
        operations: vec![NumericalOperationContract {
            operation_id: OperationId::new("operation.fixture.kv").unwrap(),
            version: ContractVersion::new(1, 0),
            multiplication_type: Some(ElementType::F32),
            accumulation_type: Some(ElementType::F32),
        }],
    }
}

#[test]
fn int8_kv_storage_charges_scales_and_preserves_the_activation_boundary() {
    let f16 = profile(false);
    let int8 = profile(true);
    int8.validate().unwrap();
    assert_eq!(
        f16.activation_type().unwrap(),
        int8.activation_type().unwrap()
    );
    let bytes = |p: &NumericalExecutionProfile| {
        p.states
            .iter()
            .map(|state| {
                state
                    .capacity_demand
                    .minimum_bytes(state.tensor.byte_len().unwrap())
                    .unwrap()
            })
            .sum::<u64>()
    };
    assert_eq!(bytes(&f16), 2048);
    assert_eq!(bytes(&int8), 1024 + 32);
    assert_ne!(f16.fingerprint().unwrap(), int8.fingerprint().unwrap());
    assert_eq!(
        int8.kv_storage_format().unwrap(),
        Some(KvStorageFormat::Int8PerTokenHeadF32ScaleV1)
    );
}

#[test]
fn int8_kv_storage_rejects_missing_wrong_or_incomplete_scale_state() {
    let valid = profile(true);
    let mut missing = valid.clone();
    missing.states.pop();
    assert!(missing.validate().is_err());
    let mut wrong_dtype = valid.clone();
    wrong_dtype.states[1].tensor.element_type = ElementType::F16;
    assert!(wrong_dtype.validate().is_err());
    let mut wrong_shape = valid.clone();
    wrong_shape.states[1].tensor.dimensions = vec![2, 2];
    assert!(wrong_shape.validate().is_err());
    let mut short = valid.clone();
    short.states[1].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 32,
        maximum_tokens: 2048,
    };
    assert!(short.validate().is_err());
    let mut partial_checkpoint = valid.clone();
    partial_checkpoint.states[1].checkpoint = StateCheckpointCapability::Unsupported;
    assert!(partial_checkpoint.validate().is_err());
    let mut padding = valid.clone();
    padding.states[1].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 64,
        maximum_tokens: 4096,
    };
    assert!(padding.validate().is_err());
    let mut overflow = valid;
    overflow.states[1].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 32,
        maximum_tokens: u64::MAX,
    };
    assert!(overflow.validate().is_err());
}

#[test]
fn kv_storage_filter_preserves_qualification_and_never_substitutes_a_format() {
    let f16 = profile(false);
    let int8 = profile(true);
    let family = f16.family_id.clone();
    let profiles = FamilyNumericalProfiles::new(
        &family,
        ContractVersion::new(1, 0),
        vec![f16.clone(), int8.clone()],
        vec![f16.id.clone(), int8.id.clone()],
    )
    .unwrap();
    assert_eq!(
        profiles
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        vec![&f16]
    );
    assert_eq!(
        profiles
            .candidates(
                &NumericalExecutionPolicy::Auto,
                KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .unwrap(),
        vec![&int8]
    );
    for (required, storage) in [
        (f16.id.clone(), KvStorageFormat::Int8PerTokenHeadF32ScaleV1),
        (int8.id.clone(), KvStorageFormat::F16),
    ] {
        assert!(profiles
            .candidates(&NumericalExecutionPolicy::Require(required), storage)
            .is_err());
    }
    let unqualified = FamilyNumericalProfiles::new(
        &family,
        ContractVersion::new(1, 0),
        vec![f16.clone(), int8.clone()],
        vec![f16.id],
    )
    .unwrap();
    assert!(unqualified
        .candidates(
            &NumericalExecutionPolicy::Auto,
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    assert!(unqualified
        .candidates(
            &NumericalExecutionPolicy::Require(int8.id),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_ok());
}

#[test]
fn kv_storage_rejects_aliases_mixed_formats_and_inapplicable_int8() {
    let mut duplicate = profile(true);
    duplicate.kv_storage.push(duplicate.kv_storage[0].clone());
    assert!(duplicate.validate().is_err());
    let mut mixed = profile(true);
    let f16 = state("f16", vec![2, 4, 128], ElementType::F16);
    mixed.kv_storage.push(KvStateStorage::F16 {
        state: f16.id.clone(),
    });
    mixed.states.push(f16.clone());
    assert!(mixed.validate().is_err());
    let mut nonsequence = profile(false);
    nonsequence.states[0].lifetime = StateLifetime::Request;
    assert!(nonsequence.validate().is_err());
    let mut no_kv = profile(false);
    no_kv.kv_storage.clear();
    no_kv.states.clear();
    let family = no_kv.family_id.clone();
    let id = no_kv.id.clone();
    let profiles =
        FamilyNumericalProfiles::new(&family, ContractVersion::new(1, 0), vec![no_kv], vec![id])
            .unwrap();
    assert_eq!(
        profiles
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap()[0]
            .kv_storage_format()
            .unwrap(),
        None
    );
    assert!(profiles
        .candidates(
            &NumericalExecutionPolicy::Auto,
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
}
