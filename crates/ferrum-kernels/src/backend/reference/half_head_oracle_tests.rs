use super::*;
use crate::backend::reference::ReferenceVNextComposition;
use ferrum_interfaces::vnext::{AttributeId, DeviceId, SemanticValue};

fn tensor(dimensions: &[u64], values: &[f32], element_type: ElementType) -> OracleTensor {
    let bytes = match element_type {
        ElementType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        ElementType::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    };
    OracleTensor::new(dimensions.to_vec(), element_type, bytes).unwrap()
}

fn attributes(hidden: u64, vocabulary: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("hidden_size").unwrap(),
            SemanticValue::Unsigned(hidden),
        ),
        (
            AttributeId::new("out_features").unwrap(),
            SemanticValue::Unsigned(vocabulary),
        ),
    ])
}

fn request(inputs: Vec<OracleTensor>, hidden: u64, vocabulary: u64) -> OperationOracleRequest {
    let contract = last_token_dense_linear_f32_f16_operands_contract().unwrap();
    OperationOracleRequest::new(
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().unwrap(),
        inputs,
        attributes(hidden, vocabulary),
    )
    .unwrap()
}

fn result(values: &[f32]) -> OperationOracleResult {
    OperationOracleResult::new(vec![tensor(
        &[1, values.len() as u64],
        values,
        ElementType::F32,
    )])
    .unwrap()
}

#[test]
fn half_operands_head_oracle_handles_cancellation_without_a_relative_or_absolute_floor() {
    let oracle = ReferenceHalfOperandsHeadOracle::new().unwrap();
    let inputs = vec![
        tensor(&[1, 4], &[16384.0, 1.0, -16384.0, -1.0], ElementType::F32),
        tensor(&[1, 4], &[16384.0, 1.0, 16384.0, 1.0], ElementType::F16),
    ];
    let request = request(inputs, 4, 1);
    let reference = oracle.invoke(&request).unwrap();
    assert_eq!(reference, result(&[0.0]));
    // Ascending F32 adds lose the +1 between the large terms and produce -1.
    // A pure-relative check against zero would reject this legal reduction.
    assert!(oracle
        .compare(&request, &result(&[-1.0]), &reference)
        .unwrap());
    assert!(!oracle
        .compare(&request, &result(&[4096.0]), &reference)
        .unwrap());

    let zeros = self::request(
        vec![
            tensor(&[1, 4], &[0.0; 4], ElementType::F32),
            tensor(&[1, 4], &[2.0; 4], ElementType::F16),
        ],
        4,
        1,
    );
    let zero_reference = oracle.invoke(&zeros).unwrap();
    assert!(oracle
        .compare(&zeros, &result(&[0.0]), &zero_reference)
        .unwrap());
    assert!(!oracle
        .compare(&zeros, &result(&[f32::from_bits(1)]), &zero_reference)
        .unwrap());
    assert!(oracle
        .compare(&request, &result(&[0.0]), &result(&[1.0]))
        .is_err());
}

#[test]
fn half_operands_head_oracle_binds_native_q6_to_one_sequence_final_row() {
    let mut coefficients = [0xff_u8; 210];
    coefficients[192..208].fill(127);
    let scale = f16::from_f32(0.001_001);
    coefficients[208..210].copy_from_slice(&scale.to_le_bytes());
    let mut activation = vec![10.0; 512];
    activation[256..].fill(0.0);
    activation[256] = 1.000_488_3;
    let inputs = q6_half_operands_oracle_inputs(
        tensor(&[2, 256], &activation, ElementType::F32),
        1,
        &coefficients,
    )
    .unwrap();
    let expected = f16::from_f32((scale.to_f32() * 127.0) * 31.0).to_f32();
    let composition =
        ReferenceVNextComposition::create(DeviceId::new("device.reference.head-oracle").unwrap())
            .unwrap();
    let registry = reference_half_operands_head_oracle_registry(composition.catalog()).unwrap();
    let operation = last_token_dense_linear_f32_f16_operands_contract().unwrap();
    let bound = registry.bind(&operation.descriptor().id).unwrap();
    assert_eq!(
        bound.invoke(inputs.clone(), attributes(256, 1)).unwrap(),
        result(&[expected])
    );
    assert!(bound
        .invoke_and_compare(inputs.clone(), attributes(256, 1), &result(&[expected]))
        .unwrap());
    assert!(!bound
        .invoke_and_compare(
            inputs.clone(),
            attributes(256, 1),
            &result(&[expected + 0.1])
        )
        .unwrap());
    let multiple_sequences =
        OperationOracleResult::new(vec![tensor(&[2, 1], &[expected; 2], ElementType::F32)])
            .unwrap();
    assert!(bound
        .invoke_and_compare(inputs, attributes(256, 1), &multiple_sequences)
        .is_err());
}

#[test]
fn half_operands_head_oracle_preserves_subnormals_and_rejects_half_overflow() {
    let oracle = ReferenceHalfOperandsHeadOracle::new().unwrap();
    let tiny = 2_f32.powi(-24);
    let request = request(
        vec![
            tensor(&[1, 1], &[tiny], ElementType::F32),
            tensor(&[1, 1], &[1.0], ElementType::F16),
        ],
        1,
        1,
    );
    let reference = oracle.invoke(&request).unwrap();
    assert_eq!(reference, result(&[tiny]));
    assert!(oracle.compare(&request, &reference, &reference).unwrap());
    assert!(
        !oracle
            .compare(&request, &result(&[0.0]), &reference)
            .unwrap(),
        "flushing a half operand must not be hidden by the error bound"
    );
    let weight_subnormal = self::request(
        vec![
            tensor(&[1, 1], &[tiny], ElementType::F32),
            tensor(&[1, 1], &[tiny], ElementType::F16),
        ],
        1,
        1,
    );
    assert_eq!(
        oracle.invoke(&weight_subnormal).unwrap(),
        result(&[2_f32.powi(-48)])
    );
    let overflow = self::request(
        vec![
            tensor(&[1, 1], &[65_520.0], ElementType::F32),
            tensor(&[1, 1], &[1.0], ElementType::F16),
        ],
        1,
        1,
    );
    assert!(oracle.invoke(&overflow).is_err());
    assert!(oracle
        .compare(&overflow, &result(&[0.0]), &result(&[0.0]))
        .is_err());
    assert!(OracleTensor::new(
        vec![1, 1],
        ElementType::F32,
        f32::NAN.to_le_bytes().to_vec()
    )
    .is_err());
    assert!(OracleTensor::new(
        vec![1, 1],
        ElementType::F32,
        f32::INFINITY.to_le_bytes().to_vec()
    )
    .is_err());

    let input = tensor(&[1, 256], &[0.0; 256], ElementType::F32);
    assert!(q6_half_operands_oracle_inputs(input.clone(), 1, &[0; 209]).is_err());
    let mut nonfinite_scale = [0xff_u8; 210];
    nonfinite_scale[208..210].copy_from_slice(&f16::INFINITY.to_le_bytes());
    assert!(q6_half_operands_oracle_inputs(input, 1, &nonfinite_scale).is_err());
    assert!(gamma(1 << 24, 2_f64.powi(-24)).is_err());
}

#[test]
fn half_operands_head_oracle_bound_covers_f64_rounding_and_checks_request_identity() {
    let oracle = ReferenceHalfOperandsHeadOracle::new().unwrap();
    // Products span 78 powers of two; naive F64 summation loses the tiny term.
    let tiny = 2_f32.powi(-24);
    let request = request(
        vec![
            tensor(&[1, 3], &[32768.0, tiny, -32768.0], ElementType::F32),
            tensor(&[1, 3], &[32768.0, tiny, 32768.0], ElementType::F16),
        ],
        3,
        1,
    );
    let dots = oracle.evaluate(&request).unwrap();
    assert_eq!(dots[0].center, 0.0);
    assert!(dots[0].error_bound > f64::from(tiny) * f64::from(tiny));
    assert!(oracle
        .compare(
            &request,
            &result(&[tiny * tiny]),
            &oracle.invoke(&request).unwrap()
        )
        .unwrap());
    let foreign = OperationOracleRequest::new(
        "operation.unrelated".to_owned().try_into().unwrap(),
        request.operation_fingerprint(),
        request.inputs().to_vec(),
        request.attributes().clone(),
    )
    .unwrap();
    assert!(oracle.invoke(&foreign).is_err());
}
