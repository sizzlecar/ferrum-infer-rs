//! Synthetic data with an exact, reduction-order-independent selected-dot oracle.
use super::*;

pub(super) const WEIGHT_SCALE: f32 = 1.0 / 256.0;

fn input_integer(row: u64, k: u64) -> i32 {
    let magnitude = if k == 0 || (row + k) % 3 == 0 {
        448
    } else {
        224
    };
    if k != 0 && (row + 3 * k) % 7 == 0 {
        -magnitude
    } else {
        magnitude
    }
}
fn weight_units(n: u64, k: u64) -> i32 {
    let magnitude = if (n + 2 * k) % 3 == 0 { 2 } else { 1 };
    if k != 0 && (3 * n + 5 * k) % 11 == 0 {
        -magnitude
    } else {
        magnitude
    }
}
fn weight_bits(n: u64, k: u64) -> u8 {
    let units = weight_units(n, k);
    (if units.abs() == 2 { 0x18 } else { 0x10 }) | if units < 0 { 0x80 } else { 0 }
}

pub(super) struct Payload {
    weights: ManagedHostBytes,
    scale: ManagedHostBytes,
}
impl Payload {
    pub fn new(shape: Shape, host: &HostMemoryAccount) -> Result<Self> {
        let e = shape.extents()?;
        let mut claims = host
            .reserve_many(&[
                (HostMemoryClass::WeightBacking, e.weights - 4),
                (HostMemoryClass::WeightBacking, 4),
            ])?
            .into_iter();
        let mut weights = claims
            .next()
            .context("missing weights claim")?
            .allocate_zeroed()?;
        let mut scale = claims
            .next()
            .context("missing scale claim")?
            .allocate_zeroed()?;
        for (index, dst) in weights.as_mut().iter_mut().enumerate() {
            *dst = weight_bits(index as u64 / shape.k, index as u64 % shape.k);
        }
        scale.as_mut().copy_from_slice(&WEIGHT_SCALE.to_le_bytes());
        Ok(Self {
            weights: weights.freeze(),
            scale: scale.freeze(),
        })
    }
}
impl WeightComponentSource for Payload {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        let bytes = match c.role {
            WeightComponentRole::PackedValues => self.weights.clone(),
            WeightComponentRole::Scales => self.scale.clone(),
            _ => return Err(invalid("unexpected synthetic weight component")),
        };
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec!["generated-finite-dyadic-benchmark-v1".into()],
            c.dimensions.clone(),
            c.physical_element_type(),
            bytes,
        )
    }
}

pub(super) fn input(shape: Shape, host: &HostMemoryAccount) -> Result<ManagedHostTensor> {
    let mut bytes =
        host.allocate_zeroed(HostMemoryClass::InputPreprocess, shape.extents()?.input)?;
    for (index, dst) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        let x = input_integer(index as u64 / shape.k, index as u64 % shape.k);
        dst.copy_from_slice(&bf16::from_f32(x as f32).to_bits().to_le_bytes());
    }
    Ok(ManagedHostTensor::new(
        id(X),
        ResolvedTensorSpec::new(
            vec![shape.m, shape.k],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?,
        bytes.freeze(),
    )?)
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Selected {
    pub row: u64,
    pub column: u64,
    pub expected_bits: u16,
}
pub(super) fn expected(shape: Shape) -> Result<Vec<Selected>> {
    shape.extents()?;
    let rows = BTreeSet::from([0, shape.m / 2, shape.m - 1]);
    let columns = BTreeSet::from([0, shape.n / 2, shape.n - 1]);
    let mut selected = Vec::new();
    for row in rows {
        for &column in &columns {
            // X*decoded(W) is an integer in [-28,28]. Every possible partial
            // sum is <=28*K<=2^24. Weight scale is an exact power of two.
            let sum = (0..shape.k)
                .map(|k| i64::from(input_integer(row, k) * weight_units(column, k) / 32))
                .sum::<i64>();
            selected.push(Selected {
                row,
                column,
                expected_bits: bf16::from_f64(sum as f64 / 256.0).to_bits(),
            });
        }
    }
    Ok(selected)
}

#[derive(Debug, Serialize)]
pub(super) struct Validation {
    pub elements: usize,
    pub nonzero_elements: usize,
    pub selected_points: usize,
}
pub(super) fn check(shape: Shape, bytes: &[u8], selected: &[Selected]) -> Result<Validation> {
    ensure!(
        bytes.len() == shape.extents()?.output,
        "output byte extent differs"
    );
    let mut nonzero = 0;
    for (index, b) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([b[0], b[1]]);
        ensure!(
            bits & 0x7f80 != 0x7f80,
            "nonfinite output at element {index}"
        );
        nonzero += usize::from(bits & 0x7fff != 0);
    }
    ensure!(nonzero > 0, "synthetic output is entirely zero");
    ensure!(!selected.is_empty(), "selected oracle is empty");
    for point in selected {
        ensure!(
            point.row < shape.m && point.column < shape.n,
            "oracle coordinate outside shape"
        );
        let offset = usize::try_from((point.row * shape.n + point.column) * 2)?;
        let actual = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        ensure!(
            actual == point.expected_bits,
            "selected [{},{}] bits {actual:04x} differ from independent {:04x}",
            point.row,
            point.column,
            point.expected_bits
        );
    }
    Ok(Validation {
        elements: bytes.len() / 2,
        nonzero_elements: nonzero,
        selected_points: selected.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pattern_has_exact_scale_and_independent_signed_products() {
        let s = Shape { m: 3, n: 32, k: 16 };
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                1024,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 96,
                    weight_backing_bytes: 516,
                    output_bytes: 192,
                    transfer_staging_bytes: 0,
                },
            )
            .unwrap(),
        );
        let input = input(s, &host).unwrap();
        let payload = Payload::new(s, &host).unwrap();
        assert_eq!(
            bf16::from_bits(u16::from_le_bytes(
                input.bytes().bytes()[..2].try_into().unwrap()
            ))
            .to_f32(),
            448.0
        );
        assert_eq!(weight_bits(0, 0), 0x18);
        assert_eq!(input_integer(0, 1) * weight_units(0, 1) / 32, 7);
        assert_eq!(input_integer(0, 7) * weight_units(0, 7) / 32, -7);
        for row in 0..s.m {
            assert_eq!(
                (0..s.k).map(|k| input_integer(row, k).abs()).max(),
                Some(448)
            );
        }
        assert_eq!(payload.scale.bytes(), WEIGHT_SCALE.to_le_bytes());
        drop((input, payload));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    #[test]
    fn predeclared_selected_bits_reject_mutation_and_nonfinite() {
        let s = Shape { m: 1, n: 16, k: 16 };
        let selected = expected(s).unwrap();
        let mut bytes = vec![0_u8; s.extents().unwrap().output];
        for point in &selected {
            let i = point.column as usize * 2;
            bytes[i..i + 2].copy_from_slice(&point.expected_bits.to_le_bytes());
        }
        assert!(check(s, &bytes, &selected).is_ok());
        let original = bytes.clone();
        let i = selected[0].column as usize * 2;
        bytes[i..i + 2].copy_from_slice(&(selected[0].expected_bits ^ 1).to_le_bytes());
        assert!(check(s, &bytes, &selected).is_err());
        bytes = original;
        bytes[2..4].copy_from_slice(&0x7f80_u16.to_le_bytes());
        assert!(check(s, &bytes, &selected).is_err());
        assert!(check(s, &bytes[..2], &selected).is_err());
    }
}
