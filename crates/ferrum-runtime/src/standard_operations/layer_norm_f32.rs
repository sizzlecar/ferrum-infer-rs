//! Centered population LayerNorm with original F32 affine parameters.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};
pub const LAYER_NORM_F32_OPERATION_ID: &str = "operation.layer_norm.f32";
pub const LAYER_NORM_F32_CAPABILITY_ID: &str = "capability.operation.layer_norm.f32";
pub const LAYER_NORM_F32_MAX_COLUMNS: u64 = 1 << 24;
/// F32 X/Y[R,H], gamma/beta[H]. For each row, mean=sum(x)/H,
/// variance=sum((x-mean)^2)/H; inverse=1/sqrt(variance+epsilon).
/// Y=((x-mean)*inverse)*gamma+beta, with separate F32 multiplication/addition
/// boundaries, no gamma offset, RMS shortcut, half conversion, implicit
/// broadcast, Welford or fused affine expression. Providers declare reduction,
/// sqrt/division and underflow profiles; no Torch binary equivalence follows.
/// NoAlias and no global scratch; static CTA/stack reduction state is bounded.
pub fn layer_norm_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut d = head_rms_norm_f32_contract()?.descriptor;
    d.id = OperationId::new(LAYER_NORM_F32_OPERATION_ID)?;
    d.provider = provider_requirement(LAYER_NORM_F32_CAPABILITY_ID, ContractVersion::new(1, 0))?;
    let matrix = vec![symbol("rows"), symbol("columns")];
    d.inputs = vec![
        contiguous_tensor(matrix.clone(), [ElementType::F32], TensorAccess::Read)?,
        contiguous_tensor(
            vec![symbol("columns")],
            [ElementType::F32],
            TensorAccess::Read,
        )?,
        contiguous_tensor(
            vec![symbol("columns")],
            [ElementType::F32],
            TensorAccess::Read,
        )?,
    ];
    d.outputs = vec![contiguous_tensor(
        matrix,
        [ElementType::F32],
        TensorAccess::Write,
    )?];
    d.attributes = AttributeSchema::new(BTreeMap::from([positive_epsilon_attribute("epsilon")?]))?;
    d.validate()?;
    Ok(StandardOperationContract { descriptor: d })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerNormF32Geometry {
    rows: u64,
    columns: u64,
    epsilon: CanonicalRational,
    elements: u64,
    arithmetic: u64,
}
impl LayerNormF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        beta: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let [rows, columns] = x.dimensions() else {
            return Err(invalid("input must be rank two"));
        };
        if x != y
            || x.element_type() != ElementType::F32
            || x.layout() != &ResolvedTensorLayout::Contiguous
            || *rows == 0
            || !(1..=LAYER_NORM_F32_MAX_COLUMNS).contains(columns)
            || gamma.dimensions() != [*columns]
            || gamma.element_type() != ElementType::F32
            || gamma.layout() != &ResolvedTensorLayout::Contiguous
            || gamma != beta
        {
            return Err(invalid("requires identical contiguous F32 X/Y[R,H], F32 gamma/beta[H], R>0 and H in 1..=2^24"));
        }
        let numerator = i128::from(epsilon.numerator());
        let denominator = i128::from(epsilon.denominator());
        if numerator * 1_000_000_000_000 < denominator || numerator > denominator {
            return Err(invalid("epsilon must be in [1e-12,1]"));
        }
        for t in [x, gamma, beta, y] {
            if t.minimum_storage_bytes()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds address space"));
            }
        }
        let elements = rows
            .checked_mul(*columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        // Fixed 256-lane implementations: 8H scalar arithmetic + 2*255 tree
        // additions, two mean/variance divisions, epsilon add, sqrt, reciprocal.
        // Logical source work, not measured GPU instructions or hardware FLOPs.
        let arithmetic = columns
            .checked_mul(8)
            .and_then(|n| n.checked_add(515))
            .and_then(|n| n.checked_mul(*rows))
            .ok_or_else(|| invalid("logical work overflows"))?;
        Ok(Self {
            rows: *rows,
            columns: *columns,
            epsilon,
            elements,
            arithmetic,
        })
    }
    pub const fn rows(self) -> u64 {
        self.rows
    }
    pub const fn columns(self) -> u64 {
        self.columns
    }
    pub const fn elements(self) -> u64 {
        self.elements
    }
    pub const fn epsilon(self) -> CanonicalRational {
        self.epsilon
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.arithmetic
    }
    pub fn effective_epsilon(self) -> f32 {
        (self.epsilon.numerator() as f64 / self.epsilon.denominator() as f64) as f32
    }
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("F32 LayerNorm: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(d: &[u64], t: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d.to_vec(), t, ResolvedTensorLayout::Contiguous).unwrap()
    }
    #[test]
    fn affine_layer_norm_f32_requires_exact_vectors_and_general_positive_width() {
        let c = layer_norm_f32_contract().unwrap();
        assert_eq!(c.descriptor().attributes.entries().len(), 1);
        let eps = CanonicalRational::new(1, 4).unwrap();
        let x = spec(&[2, 2560], ElementType::F32);
        let p = spec(&[2560], ElementType::F32);
        let g = LayerNormF32Geometry::new(&x, &p, &p, &x, eps).unwrap();
        assert_eq!(g.elements(), 5120);
        assert_eq!(g.logical_arithmetic_operations(), 2 * (2560 * 8 + 515));
        assert!(
            LayerNormF32Geometry::new(&x, &spec(&[1, 2560], ElementType::F32), &p, &x, eps)
                .is_err()
        );
        assert!(
            LayerNormF32Geometry::new(&x, &p, &spec(&[2560], ElementType::Bf16), &x, eps).is_err()
        );
        assert!(
            LayerNormF32Geometry::new(&x, &p, &p, &x, CanonicalRational::new(0, 1).unwrap())
                .is_err()
        );
        for width in [1, 3, 128, 2048] {
            let x = spec(&[1, width], ElementType::F32);
            let p = spec(&[width], ElementType::F32);
            LayerNormF32Geometry::new(&x, &p, &p, &x, eps).unwrap();
        }
    }
}
