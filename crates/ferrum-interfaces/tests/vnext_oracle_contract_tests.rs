use ferrum_interfaces::vnext::*;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn sha(character: char) -> String {
    std::iter::repeat_n(character, 64).collect()
}

fn contiguous_storage_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap()
}

fn tensor_contract(element_type: ElementType, access: TensorAccess) -> TensorContract {
    TensorContract::new(
        vec![DimensionConstraint::Exact(2)],
        BTreeSet::from([element_type]),
        vec![LayoutConstraint::Contiguous],
        access,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

fn operation(
    operation_id: &str,
    oracle: OracleSpec,
    element_type: ElementType,
) -> OperationDescriptor {
    OperationDescriptor {
        id: id(operation_id),
        version: ContractVersion::new(1, 0),
        inputs: vec![tensor_contract(element_type, TensorAccess::Read)],
        outputs: vec![tensor_contract(element_type, TensorAccess::Write)],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Forbidden,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([id("capability.compute")]),
        },
        profile_phase: ProfilePhase::Decode,
    }
}

fn catalog(operations: &[OperationDescriptor]) -> CapabilityCatalog {
    let device_id: DeviceId = id("device.oracle.reference");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let providers = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| {
            (
                operation.id.clone(),
                vec![OperationProviderDescriptor::new(
                    id(format!("provider.oracle-test.{index}")),
                    operation.id.clone(),
                    operation.fingerprint().unwrap(),
                    sha('d'),
                    ContractVersion::new(1, 0),
                    device_id.clone(),
                    capabilities.clone(),
                    BTreeSet::from([id("weight-format.oracle-test")]),
                    BTreeSet::new(),
                    vec![
                        ProviderStorageBindingRequirement::new(
                            ResolvedValueRole::Input,
                            0,
                            DynamicStorageRequirement::contiguous(),
                        ),
                        ProviderStorageBindingRequirement::new(
                            ResolvedValueRole::Output,
                            0,
                            DynamicStorageRequirement::contiguous(),
                        ),
                    ],
                    format!("resource-estimator.oracle-test.{index}"),
                    ContractVersion::new(1, 0),
                    sha('e'),
                )
                .unwrap()],
            )
        })
        .collect();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 30,
            runtime_implementation_fingerprint: sha('c'),
            capabilities: capabilities.clone(),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        operations.to_vec(),
        providers,
        vec![EngineProviderDescriptor::new(
            id("provider.engine.oracle-test"),
            ContractVersion::new(1, 0),
            sha('b'),
            device_id,
            capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

struct TestContract {
    descriptor: OperationDescriptor,
    reject_signature: bool,
}

impl OperationContract for TestContract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.descriptor
    }

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if self.reject_signature
            || inputs != self.descriptor.inputs
            || outputs != self.descriptor.outputs
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "test operation signature rejected".to_owned(),
            });
        }
        Ok(())
    }
}

fn contracts(operations: &[OperationDescriptor]) -> Vec<Box<dyn OperationContract>> {
    operations
        .iter()
        .cloned()
        .map(|descriptor| {
            Box::new(TestContract {
                descriptor,
                reject_signature: false,
            }) as Box<dyn OperationContract>
        })
        .collect()
}

fn oracle_descriptor(
    operation: &OperationDescriptor,
    oracle_id: &str,
    implementation: char,
) -> OperationOracleDescriptor {
    OperationOracleDescriptor::new(
        id(oracle_id),
        ContractVersion::new(1, 0),
        sha(implementation),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
    )
    .unwrap()
}

fn f32_tensor(values: &[f32]) -> OracleTensor {
    OracleTensor::new(
        vec![values.len() as u64],
        ElementType::F32,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn u8_tensor(values: &[u8]) -> OracleTensor {
    OracleTensor::new(vec![values.len() as u64], ElementType::U8, values.to_vec()).unwrap()
}

fn result(tensor: OracleTensor) -> OperationOracleResult {
    OperationOracleResult::new(vec![tensor]).unwrap()
}

struct EchoOracle {
    descriptor: OperationOracleDescriptor,
    calls: Arc<AtomicUsize>,
    observed_operation: Arc<Mutex<Option<OperationId>>>,
}

impl OperationOracle for EchoOracle {
    fn descriptor(&self) -> &OperationOracleDescriptor {
        &self.descriptor
    }

    fn invoke(
        &self,
        request: &OperationOracleRequest,
    ) -> Result<OperationOracleResult, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.observed_operation.lock().unwrap() = Some(request.operation_id().clone());
        if request.operation_id() != self.descriptor.operation_id()
            || request.operation_fingerprint() != self.descriptor.operation_fingerprint()
            || request.inputs().len() != 1
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "echo oracle received the wrong request identity".to_owned(),
            });
        }
        OperationOracleResult::new(vec![request.inputs()[0].clone()])
    }
}

fn echo_oracle(
    descriptor: OperationOracleDescriptor,
) -> (
    Box<dyn OperationOracle>,
    Arc<AtomicUsize>,
    Arc<Mutex<Option<OperationId>>>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(Mutex::new(None));
    (
        Box::new(EchoOracle {
            descriptor,
            calls: calls.clone(),
            observed_operation: observed.clone(),
        }),
        calls,
        observed,
    )
}

fn registration(
    operation: &OperationDescriptor,
    oracle_id: &str,
    implementation: char,
) -> (
    OperationOracleRegistration,
    Arc<AtomicUsize>,
    Arc<Mutex<Option<OperationId>>>,
) {
    let descriptor = oracle_descriptor(operation, oracle_id, implementation);
    let (oracle, calls, observed) = echo_oracle(descriptor.clone());
    (
        OperationOracleRegistration::new(descriptor, oracle).unwrap(),
        calls,
        observed,
    )
}

fn assert_invalid<T>(result: Result<T, VNextError>, expected: &str) {
    match result {
        Err(VNextError::InvalidExecutionPlan { reason }) => {
            assert!(
                reason.contains(expected),
                "expected `{expected}` in `{reason}`"
            );
        }
        Err(error) => panic!("expected InvalidExecutionPlan, got {error}"),
        Ok(_) => panic!("expected rejection containing `{expected}`"),
    }
}

#[test]
fn external_trait_object_and_registry_bound_handle_invoke() {
    let operation = operation("operation.oracle.echo", OracleSpec::Exact, ElementType::F32);
    let descriptor = oracle_descriptor(&operation, "oracle.echo.external", '1');
    let (oracle, calls, observed) = echo_oracle(descriptor.clone());
    let trait_object: &dyn OperationOracle = oracle.as_ref();
    let request = OperationOracleRequest::new(
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        vec![f32_tensor(&[1.0, 2.0])],
        BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(
        trait_object.invoke(&request).unwrap().outputs()[0],
        request.inputs()[0]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(observed.lock().unwrap().as_ref(), Some(&operation.id));

    let operations = vec![operation.clone()];
    let catalog = catalog(&operations);
    let (registration, bound_calls, _) = registration(&operation, "oracle.echo.bound", '2');
    let registry =
        OperationOracleRegistry::new(&catalog, contracts(&operations), vec![registration]).unwrap();
    let bound = registry.bind(&operation.id).unwrap();
    assert_eq!(bound.requested_operation_id(), &operation.id);
    assert_eq!(bound.terminal_operation_id(), &operation.id);
    assert_eq!(
        registry.contract(&operation.id).unwrap().descriptor(),
        &operation
    );
    let input = f32_tensor(&[3.0, 4.0]);
    assert_eq!(
        bound
            .invoke(vec![input.clone()], BTreeMap::new())
            .unwrap()
            .outputs()[0],
        input
    );
    assert_eq!(bound_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn descriptor_and_request_result_wire_require_explicit_revalidation() {
    let operation = operation("operation.oracle.wire", OracleSpec::Exact, ElementType::F32);
    let descriptor = oracle_descriptor(&operation, "oracle.wire", '3');
    let descriptor_wire =
        OperationOracleDescriptor::decode_untrusted(&serde_json::to_vec(&descriptor).unwrap())
            .unwrap();
    assert_eq!(descriptor_wire.revalidate().unwrap(), descriptor);

    let request = OperationOracleRequest::new(
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        vec![f32_tensor(&[1.0, 2.0])],
        BTreeMap::new(),
    )
    .unwrap();
    let request_wire =
        OperationOracleRequest::decode_untrusted(&serde_json::to_vec(&request).unwrap()).unwrap();
    assert_eq!(request_wire.revalidate().unwrap(), request);

    let result = result(f32_tensor(&[1.0, 2.0]));
    let result_wire =
        OperationOracleResult::decode_untrusted(&serde_json::to_vec(&result).unwrap()).unwrap();
    assert_eq!(result_wire.revalidate().unwrap(), result);

    let mut invalid_descriptor = serde_json::to_value(&descriptor).unwrap();
    invalid_descriptor["implementation_fingerprint"] = json!("not-a-sha");
    let invalid_descriptor = OperationOracleDescriptor::decode_untrusted(
        &serde_json::to_vec(&invalid_descriptor).unwrap(),
    )
    .unwrap();
    assert_invalid(
        invalid_descriptor.revalidate(),
        "implementation fingerprint",
    );

    let mut unknown_field = serde_json::to_value(&request).unwrap();
    unknown_field["caller_oracle"] = json!("oracle.impostor");
    assert!(
        OperationOracleRequest::decode_untrusted(&serde_json::to_vec(&unknown_field).unwrap())
            .is_err()
    );

    validate_oracle_wire_byte_length(MAX_ORACLE_WIRE_BYTES).unwrap();
    assert_invalid(
        validate_oracle_wire_byte_length(MAX_ORACLE_WIRE_BYTES + 1),
        "wire bytes exceed",
    );
    assert_invalid(
        OperationOracleRequest::decode_untrusted(&vec![b' '; MAX_ORACLE_WIRE_BYTES + 1]),
        "wire bytes exceed",
    );
}

#[test]
fn registry_rejects_missing_duplicate_contract_signature_and_fingerprint_mismatches() {
    let operation = operation(
        "operation.oracle.registry",
        OracleSpec::Exact,
        ElementType::F32,
    );
    let operations = vec![operation.clone()];
    let catalog = catalog(&operations);
    let (valid_registration, _, _) = registration(&operation, "oracle.registry.valid", '4');
    assert_invalid(
        OperationOracleRegistry::new(&catalog, contracts(&operations), Vec::new()),
        "exactly one oracle",
    );
    assert_invalid(
        OperationOracleRegistry::new(&catalog, Vec::new(), vec![valid_registration]),
        "exactly one contract",
    );

    let (duplicate_registration, _, _) =
        registration(&operation, "oracle.registry.duplicate-contract", '5');
    let mut duplicate_contracts = contracts(&operations);
    duplicate_contracts.extend(contracts(&operations));
    assert_invalid(
        OperationOracleRegistry::new(&catalog, duplicate_contracts, vec![duplicate_registration]),
        "duplicate contract",
    );

    let (signature_registration, _, _) = registration(&operation, "oracle.registry.signature", '6');
    let rejecting_contract: Vec<Box<dyn OperationContract>> = vec![Box::new(TestContract {
        descriptor: operation.clone(),
        reject_signature: true,
    })];
    assert_invalid(
        OperationOracleRegistry::new(&catalog, rejecting_contract, vec![signature_registration]),
        "signature rejected",
    );

    let wrong_descriptor = OperationOracleDescriptor::new(
        id("oracle.registry.wrong-fingerprint"),
        ContractVersion::new(1, 0),
        sha('7'),
        operation.id.clone(),
        sha('0'),
    )
    .unwrap();
    let (oracle, _, _) = echo_oracle(wrong_descriptor.clone());
    let wrong_registration = OperationOracleRegistration::new(wrong_descriptor, oracle).unwrap();
    assert_invalid(
        OperationOracleRegistry::new(&catalog, contracts(&operations), vec![wrong_registration]),
        "fingerprint differs",
    );

    let independent_version = OperationOracleDescriptor::new(
        id("oracle.registry.independent-version"),
        ContractVersion::new(7, 3),
        sha('7'),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
    )
    .unwrap();
    let (oracle, _, _) = echo_oracle(independent_version.clone());
    OperationOracleRegistry::new(
        &catalog,
        contracts(&operations),
        vec![OperationOracleRegistration::new(independent_version, oracle).unwrap()],
    )
    .unwrap();

    let (first, _, _) = registration(&operation, "oracle.registry.first", '8');
    let (second, _, _) = registration(&operation, "oracle.registry.second", '9');
    assert_invalid(
        OperationOracleRegistry::new(&catalog, contracts(&operations), vec![first, second]),
        "multiple oracles",
    );
}

#[test]
fn independently_anchored_descriptor_rejects_impostor_and_registry_never_accepts_call_oracle() {
    let operation = operation(
        "operation.oracle.impostor",
        OracleSpec::Exact,
        ElementType::F32,
    );
    let expected = oracle_descriptor(&operation, "oracle.impostor", 'a');
    let impostor_descriptor = OperationOracleDescriptor::new(
        expected.oracle_id().clone(),
        expected.version(),
        sha('b'),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
    )
    .unwrap();
    let (impostor, impostor_calls, _) = echo_oracle(impostor_descriptor);
    assert_invalid(
        OperationOracleRegistration::new(expected.clone(), impostor),
        "differs from its trusted registration descriptor",
    );
    assert_eq!(impostor_calls.load(Ordering::SeqCst), 0);

    let operations = vec![operation.clone()];
    let (trusted, trusted_calls, _) = echo_oracle(expected.clone());
    let registry = OperationOracleRegistry::new(
        &catalog(&operations),
        contracts(&operations),
        vec![OperationOracleRegistration::new(expected, trusted).unwrap()],
    )
    .unwrap();
    registry
        .bind(&operation.id)
        .unwrap()
        .invoke(vec![f32_tensor(&[1.0, 2.0])], BTreeMap::new())
        .unwrap();
    assert_eq!(trusted_calls.load(Ordering::SeqCst), 1);
    assert_eq!(impostor_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn reference_operation_chain_resolves_to_one_terminal_oracle() {
    let terminal = operation(
        "operation.oracle.terminal",
        OracleSpec::AbsoluteTolerance {
            tolerance: CanonicalRational::new(1, 10).unwrap(),
        },
        ElementType::F32,
    );
    let middle = operation(
        "operation.oracle.middle",
        OracleSpec::ReferenceOperation {
            operation_id: terminal.id.clone(),
            version: ContractVersion::new(1, 0),
        },
        ElementType::F32,
    );
    let head = operation(
        "operation.oracle.head",
        OracleSpec::ReferenceOperation {
            operation_id: middle.id.clone(),
            version: ContractVersion::new(1, 0),
        },
        ElementType::F32,
    );
    let operations = vec![head.clone(), middle.clone(), terminal.clone()];
    let (terminal_registration, calls, observed) =
        registration(&terminal, "oracle.reference.terminal", 'c');
    let registry = OperationOracleRegistry::new(
        &catalog(&operations),
        contracts(&operations),
        vec![terminal_registration],
    )
    .unwrap();
    let bound = registry.bind(&head.id).unwrap();
    assert_eq!(bound.requested_operation_id(), &head.id);
    assert_eq!(bound.terminal_operation_id(), &terminal.id);
    assert!(matches!(
        bound.comparison_policy(),
        OracleSpec::AbsoluteTolerance { .. }
    ));
    let actual = result(f32_tensor(&[10.05, -9.95]));
    assert!(bound
        .invoke_and_compare(vec![f32_tensor(&[10.0, -10.0])], BTreeMap::new(), &actual,)
        .unwrap());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(observed.lock().unwrap().as_ref(), Some(&terminal.id));

    let (head_registration, _, _) = registration(&head, "oracle.reference.illegal", 'd');
    assert_invalid(
        OperationOracleRegistry::new(
            &catalog(&operations),
            contracts(&operations),
            vec![
                registration(&terminal, "oracle.reference.valid", 'e').0,
                head_registration,
            ],
        ),
        "cannot register a direct oracle",
    );
}

#[test]
fn exact_absolute_and_relative_comparison_are_fail_closed() {
    let plus_zero = result(f32_tensor(&[0.0, 1.0]));
    let minus_zero = result(f32_tensor(&[-0.0, 1.0]));
    assert!(compare_oracle_results(&OracleSpec::Exact, &plus_zero, &plus_zero).unwrap());
    assert!(!compare_oracle_results(&OracleSpec::Exact, &plus_zero, &minus_zero).unwrap());

    let absolute = OracleSpec::AbsoluteTolerance {
        tolerance: CanonicalRational::new(1, 10).unwrap(),
    };
    assert!(compare_oracle_results(
        &absolute,
        &result(f32_tensor(&[1.05, -1.05])),
        &result(f32_tensor(&[1.0, -1.0])),
    )
    .unwrap());
    assert!(!compare_oracle_results(
        &absolute,
        &result(f32_tensor(&[1.2, -1.0])),
        &result(f32_tensor(&[1.0, -1.0])),
    )
    .unwrap());

    let relative = OracleSpec::RelativeTolerance {
        tolerance: CanonicalRational::new(1, 10).unwrap(),
    };
    assert!(compare_oracle_results(
        &relative,
        &result(f32_tensor(&[10.5, -10.5])),
        &result(f32_tensor(&[10.0, -10.0])),
    )
    .unwrap());
    assert!(!compare_oracle_results(
        &relative,
        &result(f32_tensor(&[11.1, -10.0])),
        &result(f32_tensor(&[10.0, -10.0])),
    )
    .unwrap());
    assert!(!compare_oracle_results(
        &relative,
        &result(f32_tensor(&[0.0001, 1.0])),
        &result(f32_tensor(&[0.0, 1.0])),
    )
    .unwrap());

    assert_invalid(
        compare_oracle_results(
            &OracleSpec::ReferenceOperation {
                operation_id: id("operation.unresolved"),
                version: ContractVersion::new(1, 0),
            },
            &plus_zero,
            &plus_zero,
        ),
        "must be resolved",
    );
    assert_invalid(
        compare_oracle_results(
            &OracleSpec::AbsoluteTolerance {
                tolerance: CanonicalRational::new(-1, 10).unwrap(),
            },
            &plus_zero,
            &plus_zero,
        ),
        "must not be negative",
    );
    assert_invalid(
        compare_oracle_results(
            &absolute,
            &result(u8_tensor(&[1, 2])),
            &result(f32_tensor(&[1.0, 2.0])),
        ),
        "shape or dtype",
    );
}

#[test]
fn host_tensor_rejects_noncanonical_nonfinite_and_overflowing_inputs() {
    OracleTensor::new(vec![1; MAX_ORACLE_TENSOR_RANK], ElementType::U8, vec![1]).unwrap();
    assert_invalid(
        OracleTensor::new(
            vec![1; MAX_ORACLE_TENSOR_RANK + 1],
            ElementType::U8,
            vec![1],
        ),
        "rank exceeds",
    );
    assert_invalid(
        OracleTensor::new(vec![0], ElementType::U8, Vec::new()),
        "zero extent",
    );
    assert_invalid(
        OracleTensor::new(
            vec![MAX_ORACLE_TENSOR_ELEMENTS as u64 + 1],
            ElementType::U8,
            Vec::new(),
        ),
        "elements exceed",
    );
    assert_invalid(
        OracleTensor::new(vec![2], ElementType::F32, vec![0; 7]),
        "exactly 8 bytes",
    );
    assert_invalid(
        OracleTensor::new(vec![2], ElementType::Bool, vec![0, 2]),
        "only 0 or 1",
    );
    assert_invalid(
        OracleTensor::new(vec![1], ElementType::F32, f32::NAN.to_le_bytes().to_vec()),
        "non-finite f32",
    );
    assert_invalid(
        OracleTensor::new(
            vec![1],
            ElementType::F32,
            f32::INFINITY.to_le_bytes().to_vec(),
        ),
        "non-finite f32",
    );
    assert_invalid(
        OracleTensor::new(vec![1], ElementType::F16, 0x7c00_u16.to_le_bytes().to_vec()),
        "non-finite f16",
    );
    assert_invalid(
        OracleTensor::new(
            vec![1],
            ElementType::Bf16,
            0x7f80_u16.to_le_bytes().to_vec(),
        ),
        "non-finite bf16",
    );
    assert_invalid(
        OracleTensor::new(vec![u64::MAX, 2], ElementType::U8, Vec::new()),
        "overflows usize",
    );
}

#[test]
fn request_result_count_and_attribute_bounds_are_enforced() {
    let operation_id: OperationId = id("operation.oracle.bounds");
    let scalar = OracleTensor::new(Vec::new(), ElementType::U8, vec![1]).unwrap();
    OperationOracleRequest::new(
        operation_id.clone(),
        sha('f'),
        vec![scalar.clone(); MAX_ORACLE_TENSORS],
        BTreeMap::new(),
    )
    .unwrap();
    assert_invalid(
        OperationOracleRequest::new(
            operation_id.clone(),
            sha('f'),
            vec![scalar.clone(); MAX_ORACLE_TENSORS + 1],
            BTreeMap::new(),
        ),
        "0 to 64 tensors",
    );
    OperationOracleResult::new(vec![scalar.clone(); MAX_ORACLE_TENSORS]).unwrap();
    assert_invalid(OperationOracleResult::new(Vec::new()), "1 to 64 tensors");
    assert_invalid(
        OperationOracleResult::new(vec![scalar.clone(); MAX_ORACLE_TENSORS + 1]),
        "1 to 64 tensors",
    );

    assert_eq!(
        MAX_ORACLE_TENSOR_ELEMENTS * ElementType::F32.size_bytes() as usize,
        MAX_ORACLE_CALL_BYTES
    );
    let max_tensor = OracleTensor::new(
        vec![MAX_ORACLE_TENSOR_ELEMENTS as u64],
        ElementType::F32,
        vec![0; MAX_ORACLE_TENSOR_BYTES],
    )
    .unwrap();
    let max_request = OperationOracleRequest::new(
        operation_id.clone(),
        sha('f'),
        vec![max_tensor],
        BTreeMap::new(),
    )
    .unwrap();
    drop(max_request);
    let max_tensor = OracleTensor::new(
        vec![MAX_ORACLE_TENSOR_ELEMENTS as u64],
        ElementType::F32,
        vec![0; MAX_ORACLE_TENSOR_BYTES],
    )
    .unwrap();
    assert_invalid(
        OperationOracleRequest::new(
            operation_id.clone(),
            sha('f'),
            vec![max_tensor, scalar.clone()],
            BTreeMap::new(),
        ),
        "cumulative bytes",
    );

    let max_attributes = (0..MAX_ORACLE_ATTRIBUTES)
        .map(|index| (id(format!("attribute.{index}")), SemanticValue::Bool(true)))
        .collect();
    OperationOracleRequest::new(operation_id.clone(), sha('f'), Vec::new(), max_attributes)
        .unwrap();
    let over_attributes = (0..=MAX_ORACLE_ATTRIBUTES)
        .map(|index| (id(format!("attribute.{index}")), SemanticValue::Bool(true)))
        .collect();
    assert_invalid(
        OperationOracleRequest::new(operation_id.clone(), sha('f'), Vec::new(), over_attributes),
        "exceeds 256 attributes",
    );
    assert_invalid(
        OperationOracleRequest::new(
            operation_id,
            sha('f'),
            Vec::new(),
            BTreeMap::from([(
                id("attribute.large"),
                SemanticValue::Text("x".repeat(MAX_ORACLE_ATTRIBUTE_BYTES + 1)),
            )]),
        ),
        "canonical bytes",
    );
}

#[test]
fn operation_oracle_contract_proof_line() {
    const EXPECTED: usize = 26;
    let mut passed = 0usize;
    macro_rules! check {
        ($condition:expr) => {{
            assert!($condition);
            passed += 1;
        }};
    }

    let base_operation = operation(
        "operation.oracle.proof",
        OracleSpec::Exact,
        ElementType::F32,
    );
    let bool_tolerance = operation(
        "operation.oracle.bool-tolerance",
        OracleSpec::AbsoluteTolerance {
            tolerance: CanonicalRational::new(1, 10).unwrap(),
        },
        ElementType::Bool,
    );
    check!(bool_tolerance.validate().is_err());
    let descriptor = oracle_descriptor(&base_operation, "oracle.proof", '1');
    check!(descriptor.operation_id() == &base_operation.id);
    check!(descriptor.operation_fingerprint() == base_operation.fingerprint().unwrap());
    check!(
        OperationOracleDescriptor::decode_untrusted(&serde_json::to_vec(&descriptor).unwrap())
            .unwrap()
            .revalidate()
            .unwrap()
            == descriptor
    );

    check!(OracleTensor::new(vec![1; MAX_ORACLE_TENSOR_RANK], ElementType::U8, vec![1],).is_ok());
    check!(OracleTensor::new(vec![2], ElementType::F32, vec![0; 7]).is_err());
    check!(OracleTensor::new(vec![1], ElementType::F32, f32::NAN.to_le_bytes().to_vec(),).is_err());
    check!(OracleTensor::new(
        vec![1; MAX_ORACLE_TENSOR_RANK + 1],
        ElementType::U8,
        vec![1],
    )
    .is_err());

    let scalar = OracleTensor::new(Vec::new(), ElementType::U8, vec![1]).unwrap();
    check!(OperationOracleRequest::new(
        id("operation.oracle.proof-count"),
        sha('2'),
        vec![scalar.clone(); MAX_ORACLE_TENSORS],
        BTreeMap::new(),
    )
    .is_ok());
    check!(OperationOracleRequest::new(
        id("operation.oracle.proof-count"),
        sha('2'),
        vec![scalar.clone(); MAX_ORACLE_TENSORS + 1],
        BTreeMap::new(),
    )
    .is_err());
    check!(OperationOracleResult::new(Vec::new()).is_err());

    let exact = result(f32_tensor(&[0.0, 1.0]));
    check!(compare_oracle_results(&OracleSpec::Exact, &exact, &exact).unwrap());
    check!(!compare_oracle_results(
        &OracleSpec::Exact,
        &exact,
        &result(f32_tensor(&[-0.0, 1.0])),
    )
    .unwrap());
    let absolute = OracleSpec::AbsoluteTolerance {
        tolerance: CanonicalRational::new(1, 10).unwrap(),
    };
    check!(compare_oracle_results(
        &absolute,
        &result(f32_tensor(&[1.05, 2.0])),
        &result(f32_tensor(&[1.0, 2.0])),
    )
    .unwrap());
    check!(!compare_oracle_results(
        &absolute,
        &result(f32_tensor(&[1.2, 2.0])),
        &result(f32_tensor(&[1.0, 2.0])),
    )
    .unwrap());
    let relative = OracleSpec::RelativeTolerance {
        tolerance: CanonicalRational::new(1, 10).unwrap(),
    };
    check!(compare_oracle_results(
        &relative,
        &result(f32_tensor(&[10.5, 2.0])),
        &result(f32_tensor(&[10.0, 2.0])),
    )
    .unwrap());
    check!(!compare_oracle_results(
        &relative,
        &result(f32_tensor(&[11.5, 2.0])),
        &result(f32_tensor(&[10.0, 2.0])),
    )
    .unwrap());

    check!(validate_oracle_wire_byte_length(MAX_ORACLE_WIRE_BYTES).is_ok());
    check!(validate_oracle_wire_byte_length(MAX_ORACLE_WIRE_BYTES + 1).is_err());

    let operations = vec![base_operation.clone()];
    let capability_catalog = catalog(&operations);
    check!(
        OperationOracleRegistry::new(&capability_catalog, contracts(&operations), Vec::new(),)
            .is_err()
    );
    let (trusted_registration, trusted_calls, _) =
        registration(&base_operation, "oracle.proof.trusted", '3');
    let registry = OperationOracleRegistry::new(
        &capability_catalog,
        contracts(&operations),
        vec![trusted_registration],
    )
    .unwrap();
    check!(registry.catalog_fingerprint().len() == 64);
    let output = registry
        .bind(&base_operation.id)
        .unwrap()
        .invoke(vec![f32_tensor(&[3.0, 4.0])], BTreeMap::new())
        .unwrap();
    check!(output == result(f32_tensor(&[3.0, 4.0])));
    check!(trusted_calls.load(Ordering::SeqCst) == 1);

    let expected = oracle_descriptor(&base_operation, "oracle.proof.impostor", '4');
    let impostor_descriptor = OperationOracleDescriptor::new(
        expected.oracle_id().clone(),
        expected.version(),
        sha('5'),
        base_operation.id.clone(),
        base_operation.fingerprint().unwrap(),
    )
    .unwrap();
    let (impostor, _, _) = echo_oracle(impostor_descriptor);
    check!(OperationOracleRegistration::new(expected, impostor).is_err());

    let terminal = operation(
        "operation.oracle.proof-terminal",
        OracleSpec::Exact,
        ElementType::F32,
    );
    let head = operation(
        "operation.oracle.proof-head",
        OracleSpec::ReferenceOperation {
            operation_id: terminal.id.clone(),
            version: ContractVersion::new(1, 0),
        },
        ElementType::F32,
    );
    let reference_operations = vec![head.clone(), terminal.clone()];
    let reference_registry = OperationOracleRegistry::new(
        &catalog(&reference_operations),
        contracts(&reference_operations),
        vec![registration(&terminal, "oracle.proof.terminal", '6').0],
    )
    .unwrap();
    let bound = reference_registry.bind(&head.id).unwrap();
    check!(bound.terminal_operation_id() == &terminal.id);
    check!(bound
        .invoke_and_compare(
            vec![f32_tensor(&[7.0, 8.0])],
            BTreeMap::new(),
            &result(f32_tensor(&[7.0, 8.0])),
        )
        .unwrap());

    assert_eq!(passed, EXPECTED);
    println!("\nVNEXT OPERATION ORACLE PASS: {passed}/{EXPECTED}");
}
