//! Request-bound numerical oracle for the explicit half-operands output head.
//!
//! One request is one sequence, with a `[tokens, hidden]` activation and a
//! `[vocabulary, hidden]` logical F16 weight. It produces `[1, vocabulary]` from
//! the final token. Batched provider conformance must submit each participant's
//! own token span; a packed batch is not a single oracle sequence.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    last_token_dense_linear_f32_f16_operands_contract, CapabilityCatalog, ContractVersion,
    ElementType, OperationContract, OperationOracle, OperationOracleDescriptor, OperationOracleId,
    OperationOracleRegistration, OperationOracleRegistry, OperationOracleRequest,
    OperationOracleResult, OracleTensor, VNextError, MAX_ORACLE_TENSOR_ELEMENTS,
};
use half::f16;

use crate::gguf_blocks::GgufBlockFormat;

use super::dense_linear::{dimension, implementation_fingerprint};

/// Independent F64 center and condition-scaled F32 reduction error bound.
pub struct ReferenceHalfOperandsHeadOracle {
    descriptor: OperationOracleDescriptor,
}

impl ReferenceHalfOperandsHeadOracle {
    pub fn new() -> Result<Self, VNextError> {
        let contract = last_token_dense_linear_f32_f16_operands_contract()?;
        let descriptor = OperationOracleDescriptor::new(
            OperationOracleId::new("oracle.reference.last-token.f32.f16-operands")?,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[
                include_str!("half_head_oracle.rs").as_bytes(),
                include_str!("../../gguf_blocks/mod.rs").as_bytes(),
            ]),
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint()?,
        )?;
        Ok(Self { descriptor })
    }

    pub fn registration(self) -> Result<OperationOracleRegistration, VNextError> {
        OperationOracleRegistration::new(self.descriptor.clone(), Box::new(self))
    }

    fn evaluate(&self, request: &OperationOracleRequest) -> Result<Vec<DotReference>, VNextError> {
        if request.operation_id() != self.descriptor.operation_id()
            || request.operation_fingerprint() != self.descriptor.operation_fingerprint()
        {
            return Err(invalid(
                "half-operands oracle request belongs to another operation",
            ));
        }
        let contract = last_token_dense_linear_f32_f16_operands_contract()?;
        contract
            .descriptor()
            .validate_attributes(request.attributes())?;
        let hidden = dimension(request.attributes(), "hidden_size").map_err(invalid)?;
        let vocabulary = dimension(request.attributes(), "out_features").map_err(invalid)?;
        let [input, weight] = request.inputs() else {
            return Err(invalid(
                "half-operands oracle requires activation and logical F16 weight",
            ));
        };
        if input.element_type() != ElementType::F32
            || input.dimensions().len() != 2
            || input.dimensions()[1] != hidden as u64
            || weight.element_type() != ElementType::F16
            || weight.dimensions() != [vocabulary as u64, hidden as u64]
        {
            return Err(invalid(
                "half-operands oracle requires one sequence and one weight matrix",
            ));
        }
        let selected = input
            .element_count()
            .checked_sub(hidden)
            .ok_or_else(|| invalid("half-operands oracle has no final token"))?;
        let rounded_input = input.bytes()[selected * 4..]
            .chunks_exact(4)
            .map(|bytes| finite_half(f32::from_le_bytes(bytes.try_into().unwrap())))
            .collect::<Result<Vec<_>, _>>()?;
        let gamma32 = gamma(hidden, 2_f64.powi(-24))?;
        let gamma64 = gamma(hidden, 2_f64.powi(-53))?;
        let mut references = Vec::with_capacity(vocabulary);
        for row in weight.bytes().chunks_exact(hidden * 2) {
            let mut center = 0.0_f64;
            let mut sum_abs = 0.0_f64;
            for (activation, bytes) in rounded_input.iter().zip(row.chunks_exact(2)) {
                let coefficient = f16::from_le_bytes(bytes.try_into().unwrap()).to_f64();
                // Products of finite half operands are exact in both F32 and F64.
                // Even minimum-half subnormal products (2^-48) are normal F32.
                let product = activation * coefficient;
                center += product;
                sum_abs += product.abs();
            }
            if !center.is_finite() || !sum_abs.is_finite() {
                return Err(invalid(
                    "half-operands oracle produced non-finite reference arithmetic",
                ));
            }
            // F64 accumulation is not assumed exact. Inflate the computed sum
            // of absolute products and both reduction bounds outward.
            let error_bound = if sum_abs == 0.0 {
                0.0
            } else {
                let sum_abs_upper = (sum_abs / (1.0 - gamma64).next_down()).next_up();
                ((gamma32 + gamma64).next_up() * sum_abs_upper).next_up()
            };
            references.push(DotReference {
                center,
                error_bound,
            });
        }
        Ok(references)
    }
}

impl OperationOracle for ReferenceHalfOperandsHeadOracle {
    fn descriptor(&self) -> &OperationOracleDescriptor {
        &self.descriptor
    }

    fn invoke(
        &self,
        request: &OperationOracleRequest,
    ) -> Result<OperationOracleResult, VNextError> {
        reference_result(&self.evaluate(request)?)
    }

    fn compare(
        &self,
        request: &OperationOracleRequest,
        actual: &OperationOracleResult,
        reference: &OperationOracleResult,
    ) -> Result<bool, VNextError> {
        let dots = self.evaluate(request)?;
        if reference != &reference_result(&dots)? {
            return Err(invalid(
                "half-operands comparison reference differs from this request",
            ));
        }
        let [output] = actual.outputs() else {
            return Err(invalid("half-operands oracle expects one output tensor"));
        };
        if output.element_type() != ElementType::F32
            || output.dimensions() != [1, dots.len() as u64]
        {
            return Err(invalid(
                "half-operands comparison output must be one F32 logit row",
            ));
        }
        Ok(output
            .bytes()
            .chunks_exact(4)
            .zip(dots)
            .all(|(bytes, dot)| {
                let actual = f64::from(f32::from_le_bytes(bytes.try_into().unwrap()));
                let error = (actual - dot.center).abs();
                actual.is_finite() && (error == 0.0 || error.next_up() <= dot.error_bound)
            }))
    }
}

struct DotReference {
    center: f64,
    error_bound: f64,
}

fn gamma(length: usize, unit_roundoff: f64) -> Result<f64, VNextError> {
    let product = length as f64 * unit_roundoff;
    if length == 0 || product >= 1.0 {
        return Err(invalid(
            "half-operands oracle reduction length has no finite gamma bound",
        ));
    }
    Ok((product / (1.0 - product).next_down()).next_up())
}

fn reference_result(dots: &[DotReference]) -> Result<OperationOracleResult, VNextError> {
    // The serialized reference uses F32 logits, but comparisons above use the
    // F64 center directly and include its accumulation uncertainty.
    OperationOracleResult::new(vec![OracleTensor::new(
        vec![1, dots.len() as u64],
        ElementType::F32,
        dots.iter()
            .flat_map(|dot| (dot.center as f32).to_le_bytes())
            .collect(),
    )?])
}

fn finite_half(value: f32) -> Result<f64, VNextError> {
    let rounded = f16::from_f32(value);
    if !rounded.is_finite() {
        return Err(invalid(
            "half-operands numerical oracle rejects non-finite or overflowing F16 operands",
        ));
    }
    Ok(rounded.to_f64())
}

/// Convert native Q6_K blocks into the operation's canonical logical F16 input.
/// Uses the scalar decoder independently of the reference provider's block
/// decoder; malformed spans, half overflow and non-finite scales fail closed.
pub fn q6_half_operands_oracle_inputs(
    activation: OracleTensor,
    out_features: u64,
    native_q6: &[u8],
) -> Result<Vec<OracleTensor>, VNextError> {
    if activation.element_type() != ElementType::F32 || activation.dimensions().len() != 2 {
        return Err(invalid(
            "Q6 head oracle activation must be [tokens, hidden] F32",
        ));
    }
    let hidden = usize::try_from(activation.dimensions()[1])
        .map_err(|_| invalid("Q6 hidden size exceeds usize"))?;
    let vocabulary =
        usize::try_from(out_features).map_err(|_| invalid("Q6 vocabulary exceeds usize"))?;
    let elements = hidden
        .checked_mul(vocabulary)
        .filter(|count| *count > 0 && *count <= MAX_ORACLE_TENSOR_ELEMENTS)
        .ok_or_else(|| invalid("Q6 head oracle weight exceeds the bounded tensor domain"))?;
    if !hidden.is_multiple_of(256) || (elements / 256).checked_mul(210) != Some(native_q6.len()) {
        return Err(invalid(
            "Q6 head oracle requires complete exact native blocks",
        ));
    }
    let mut weights = Vec::with_capacity(elements * 2);
    for block in native_q6.chunks_exact(210) {
        for index in 0..256 {
            let coefficient = GgufBlockFormat::Q6K.decode_value(block, index);
            finite_half(coefficient)?;
            weights.extend_from_slice(&f16::from_f32(coefficient).to_le_bytes());
        }
    }
    Ok(vec![
        activation,
        OracleTensor::new(vec![out_features, hidden as u64], ElementType::F16, weights)?,
    ])
}

/// Build an explicitly head-only conformance registry from a provider catalog.
/// Other operations remain outside this registry; no generic oracle is inferred.
pub fn reference_half_operands_head_oracle_registry(
    catalog: &CapabilityCatalog,
) -> Result<OperationOracleRegistry, VNextError> {
    let contract = last_token_dense_linear_f32_f16_operands_contract()?;
    let operation = contract.descriptor().clone();
    let providers = catalog.providers_for(&operation.id)?.to_vec();
    let scoped = CapabilityCatalog::new(
        catalog.device().clone(),
        vec![operation.clone()],
        BTreeMap::from([(operation.id, providers)]),
        catalog.engine_providers().values().cloned().collect(),
    )?;
    OperationOracleRegistry::new(
        &scoped,
        vec![Box::new(contract)],
        vec![ReferenceHalfOperandsHeadOracle::new()?.registration()?],
    )
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "half_head_oracle_tests.rs"]
mod tests;
