use super::*;

struct ObservedComparison {
    request: OperationOracleRequest,
    actual: OperationOracleResult,
    reference: OperationOracleResult,
}

struct RequestComparingOracle {
    descriptor: OperationOracleDescriptor,
    changed_descriptor: Option<OperationOracleDescriptor>,
    comparisons: Arc<AtomicUsize>,
    observed: Arc<Mutex<Option<ObservedComparison>>>,
}

impl OperationOracle for RequestComparingOracle {
    fn descriptor(&self) -> &OperationOracleDescriptor {
        if self.comparisons.load(Ordering::SeqCst) > 0 {
            if let Some(changed) = &self.changed_descriptor {
                return changed;
            }
        }
        &self.descriptor
    }

    fn invoke(
        &self,
        request: &OperationOracleRequest,
    ) -> Result<OperationOracleResult, VNextError> {
        OperationOracleResult::new(vec![request.inputs()[0].clone()])
    }

    fn compare(
        &self,
        request: &OperationOracleRequest,
        actual: &OperationOracleResult,
        reference: &OperationOracleResult,
    ) -> Result<bool, VNextError> {
        self.comparisons.fetch_add(1, Ordering::SeqCst);
        *self.observed.lock().unwrap() = Some(ObservedComparison {
            request: request.clone(),
            actual: actual.clone(),
            reference: reference.clone(),
        });
        Ok(actual == reference)
    }
}

fn request_registration(
    operation: &OperationDescriptor,
    change_descriptor: bool,
) -> (
    OperationOracleRegistration,
    Arc<AtomicUsize>,
    Arc<Mutex<Option<ObservedComparison>>>,
) {
    let descriptor = oracle_descriptor(operation, "oracle.request-comparison", '1');
    let changed_descriptor =
        change_descriptor.then(|| oracle_descriptor(operation, "oracle.changed-comparison", '2'));
    let comparisons = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(Mutex::new(None));
    let oracle = RequestComparingOracle {
        descriptor: descriptor.clone(),
        changed_descriptor,
        comparisons: Arc::clone(&comparisons),
        observed: Arc::clone(&observed),
    };
    (
        OperationOracleRegistration::new(descriptor, Box::new(oracle)).unwrap(),
        comparisons,
        observed,
    )
}

#[test]
fn operation_defined_comparison_requires_registered_implementation_and_request() {
    let operation = operation(
        "operation.context-required",
        OracleSpec::OperationDefined,
        ElementType::F32,
    );
    let (registration, _, _) = registration(&operation, "oracle.default-rejection", 'a');
    let operations = [operation.clone()];
    let registry = OperationOracleRegistry::new(
        &catalog(&operations),
        contracts(&operations),
        vec![registration],
    )
    .unwrap();
    let input = f32_tensor(&[0.0, 1.0]);
    let actual = result(input.clone());
    assert_invalid(
        registry.bind(&operation.id).unwrap().invoke_and_compare(
            vec![input],
            BTreeMap::new(),
            &actual,
        ),
        "operation-defined comparison is not implemented",
    );
    assert_invalid(
        compare_oracle_results(&OracleSpec::OperationDefined, &actual, &actual),
        "requires a registry-bound oracle and validated request",
    );
    let wire = serde_json::to_value(&operation).unwrap();
    assert_eq!(wire["oracle"], json!("operation_defined"));
    let decoded: OperationDescriptor = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded, operation);
    let mut old = operation.clone();
    old.oracle = OracleSpec::Exact;
    assert_ne!(operation.fingerprint().unwrap(), old.fingerprint().unwrap());
}

#[test]
fn operation_defined_comparison_receives_validated_terminal_context() {
    let mut terminal = operation(
        "operation.context-terminal",
        OracleSpec::OperationDefined,
        ElementType::F32,
    );
    terminal.attributes = AttributeSchema::new(BTreeMap::from([(
        id("reduction_width"),
        AttributeSpec {
            value_kind: AttributeValueKind::Unsigned,
            required: true,
            constraint: AttributeConstraint::None,
        },
    )]))
    .unwrap();
    let mut requested = terminal.clone();
    requested.id = id("operation.context-reference");
    requested.oracle = OracleSpec::ReferenceOperation {
        operation_id: terminal.id.clone(),
        version: terminal.version,
    };
    let operations = [terminal.clone(), requested.clone()];
    let (registration, calls, observed) = request_registration(&terminal, false);
    let registry = OperationOracleRegistry::new(
        &catalog(&operations),
        contracts(&operations),
        vec![registration],
    )
    .unwrap();
    let bound = registry.bind(&requested.id).unwrap();
    let input = f32_tensor(&[2.0, -3.0]);
    let attrs = BTreeMap::from([(id("reduction_width"), SemanticValue::Unsigned(256))]);
    let actual = result(input.clone());
    assert!(bound
        .invoke_and_compare(vec![input.clone()], attrs.clone(), &actual)
        .unwrap());
    {
        let observed = observed.lock().unwrap();
        let observed = observed.as_ref().unwrap();
        assert_eq!(observed.request.operation_id(), &terminal.id);
        assert_eq!(
            observed.request.operation_fingerprint(),
            terminal.fingerprint().unwrap()
        );
        assert_eq!(observed.request.inputs(), &[input.clone()]);
        assert_eq!(observed.request.attributes(), &attrs);
        assert_eq!(observed.actual, actual);
        assert_eq!(observed.reference, actual);
    }
    assert!(!bound
        .invoke_and_compare(
            vec![input.clone()],
            attrs.clone(),
            &result(f32_tensor(&[2.0, 4.0]))
        )
        .unwrap());
    let calls_before_invalid = calls.load(Ordering::SeqCst);
    assert!(bound
        .invoke_and_compare(
            vec![input.clone()],
            attrs.clone(),
            &result(f32_tensor(&[2.0]))
        )
        .is_err());
    assert!(bound
        .invoke_and_compare(vec![f32_tensor(&[2.0])], attrs.clone(), &actual)
        .is_err());
    assert!(bound
        .invoke_and_compare(vec![input], BTreeMap::new(), &actual)
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), calls_before_invalid);
}

#[test]
fn operation_defined_comparison_rechecks_oracle_identity_after_callback() {
    let operation = operation(
        "operation.context-identity",
        OracleSpec::OperationDefined,
        ElementType::F32,
    );
    let operations = [operation.clone()];
    let (registration, _, _) = request_registration(&operation, true);
    let registry = OperationOracleRegistry::new(
        &catalog(&operations),
        contracts(&operations),
        vec![registration],
    )
    .unwrap();
    let input = f32_tensor(&[2.0, -3.0]);
    assert_invalid(
        registry.bind(&operation.id).unwrap().invoke_and_compare(
            vec![input.clone()],
            BTreeMap::new(),
            &result(input),
        ),
        "descriptor changed during comparison",
    );
    assert_invalid(
        registry.bind(&operation.id),
        "descriptor changed before binding",
    );
}

#[test]
fn existing_comparison_policies_do_not_invoke_operation_defined_callback() {
    for policy in [
        OracleSpec::Exact,
        OracleSpec::AbsoluteTolerance {
            tolerance: CanonicalRational::new(1, 10).unwrap(),
        },
        OracleSpec::RelativeTolerance {
            tolerance: CanonicalRational::new(1, 10).unwrap(),
        },
    ] {
        let operation = operation("operation.context-old-policy", policy, ElementType::F32);
        let operations = [operation.clone()];
        let (registration, calls, observed) = request_registration(&operation, true);
        let registry = OperationOracleRegistry::new(
            &catalog(&operations),
            contracts(&operations),
            vec![registration],
        )
        .unwrap();
        let input = f32_tensor(&[2.0, -3.0]);
        assert!(registry
            .bind(&operation.id)
            .unwrap()
            .invoke_and_compare(vec![input.clone()], BTreeMap::new(), &result(input))
            .unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(observed.lock().unwrap().is_none());
    }
}
