//! Bounded, test-only inspection of the existing signed integer witness.
//! This never supplies prediction authority and never changes a frozen hint.
use super::*;
use serde_json::{json, Value};

const MAX_REMAINDER_DETAILS: usize = 4;

#[derive(Clone)]
struct Remainder {
    axis: usize,
    input: u64,
    fit_maximum: u64,
    scaled_remainder: i128,
    scaled_correction: u128,
    cap: AxisCap,
}

impl Remainder {
    fn value(&self) -> Value {
        json!({
            "axis": self.axis,
            "input_value": self.input,
            "fit_maximum": self.fit_maximum,
            "scaled_positive_remainder": self.scaled_remainder.to_string(),
            "scaled_correction": self.scaled_correction.to_string(),
            "correction_ns_ceiling": certificates::ceil_div(
                self.scaled_correction, WEIGHT_DENOMINATOR as u128
            ).ok().map(|v| v.to_string()),
            "coefficient_cap_numerator_ns": self.cap.numerator.to_string(),
            "coefficient_cap_denominator": self.cap.denominator,
        })
    }
}

impl SignedBasis {
    pub(in super::super) fn diagnose_bound(
        &self,
        input: &[u64],
        maxima: &[u64],
        epsilon_ns: u64,
        frozen: &SignedBasisCertificateV1,
    ) -> Value {
        let mut result = json!({
            "denominator": WEIGHT_DENOMINATOR as u64,
            "anchor_count": self.anchors.len(),
            "upper_ns": null,
            "failure": null,
            "failure_axis": null,
            "failure_anchor": null,
            "boundary": "test-only exact witness decomposition; floating weights are advice, not authority",
        });
        let mut weights = Vec::new();
        let mut top = Vec::<Remainder>::with_capacity(MAX_REMAINDER_DETAILS);
        let mut positive_axes = 0usize;
        let mut correction_sum = Some(0u128);
        let mut failure_axis = None;
        let mut failure_anchor = None;
        // Only the original candidate's operations decide success. Optional
        // explanatory sums may overflow independently and are then null.
        let calculated = (|| -> std::result::Result<u64, &'static str> {
            let dims = input.len();
            if dims == 0
                || maxima.len() != dims
                || self.caps.len() != dims
                || frozen.normalized_query_to_anchor_bits.len()
                    != self
                        .anchors
                        .len()
                        .checked_mul(dims)
                        .ok_or("dimension_overflow")?
            {
                return Err("dimensions");
            }
            weights = vector::<i64>(self.anchors.len()).map_err(|_| "weight_allocation")?;
            for (anchor, mapping) in frozen
                .normalized_query_to_anchor_bits
                .chunks_exact(dims)
                .enumerate()
            {
                failure_anchor = Some(anchor);
                let mut weight = 0.0;
                for ((&bits, &x), &scale) in mapping.iter().zip(input).zip(maxima) {
                    weight += f64::from_bits(bits) * (x as f64 / scale.max(1) as f64);
                }
                let fixed = (weight * WEIGHT_DENOMINATOR as f64).round();
                let endpoint = (1_u64 << 63) as f64;
                if !fixed.is_finite() {
                    return Err("nonfinite_weight");
                }
                if fixed < -endpoint || fixed >= endpoint {
                    return Err("weight_out_of_i64_range");
                }
                weights.push(fixed as i64);
            }
            failure_anchor = None;
            let l1 = weights.iter().try_fold(0u128, |sum, &w| {
                sum.checked_add(u128::from(w.unsigned_abs()))
            });
            result["weight_l1_scaled"] = json!(l1.map(|v| v.to_string()));
            result["weight_l1"] = json!(l1.map(|v| v as f64 / WEIGHT_DENOMINATOR as f64));
            result["negative_weights"] = json!(weights.iter().filter(|w| **w < 0).count());
            result["nonzero_weights"] = json!(weights.iter().filter(|w| **w != 0).count());
            let epsilon_allowance = l1.and_then(|v| v.checked_mul(u128::from(epsilon_ns)));
            result["epsilon_allowance_scaled"] = json!(epsilon_allowance.map(|v| v.to_string()));
            result["epsilon_allowance_ns_ceiling"] = json!(epsilon_allowance
                .and_then(|v| certificates::ceil_div(v, WEIGHT_DENOMINATOR as u128).ok())
                .map(|v| v.to_string()));
            let wall_sum =
                self.anchors
                    .iter()
                    .zip(&weights)
                    .try_fold(0i128, |sum, (anchor, &weight)| {
                        sum.checked_add(i128::from(weight).checked_mul(i128::from(anchor.wall_ns))?)
                    });
            result["anchor_wall_sum_scaled"] = json!(wall_sum.map(|v| v.to_string()));
            let mut bound = 0i128;
            for (index, (anchor, &weight)) in self.anchors.iter().zip(&weights).enumerate() {
                failure_anchor = Some(index);
                let wall = i128::from(anchor.wall_ns);
                let epsilon = i128::from(epsilon_ns);
                let endpoint = if weight >= 0 {
                    wall.checked_add(epsilon)
                } else {
                    wall.checked_sub(epsilon)
                }
                .ok_or("anchor_endpoint_overflow")?;
                bound = bound
                    .checked_add(
                        i128::from(weight)
                            .checked_mul(endpoint)
                            .ok_or("anchor_product_overflow")?,
                    )
                    .ok_or("anchor_sum_overflow")?;
            }
            failure_anchor = None;
            result["anchor_bound_scaled"] = json!(bound.to_string());
            for (axis, (&query, cap)) in input.iter().zip(&self.caps).enumerate() {
                failure_axis = Some(axis);
                let mut actual = 0i128;
                for (anchor, &weight) in self.anchors.iter().zip(&weights) {
                    let value = anchor.axes.get(axis).ok_or("anchor_dimensions")?;
                    actual = actual
                        .checked_add(
                            i128::from(weight)
                                .checked_mul(i128::from(*value))
                                .ok_or("reconstruction_product_overflow")?,
                        )
                        .ok_or("reconstruction_sum_overflow")?;
                }
                let remainder = i128::from(query)
                    .checked_mul(WEIGHT_DENOMINATOR)
                    .ok_or("query_scaling_overflow")?
                    .checked_sub(actual)
                    .ok_or("remainder_overflow")?;
                if remainder <= 0 {
                    continue;
                }
                positive_axes += 1;
                if cap.denominator == 0 {
                    return Err("unobserved_positive_remainder_axis");
                }
                let product = u128::try_from(remainder)
                    .map_err(|_| "negative_remainder")?
                    .checked_mul(cap.numerator)
                    .ok_or("coefficient_cap_product_overflow")?;
                let correction = certificates::ceil_div(product, u128::from(cap.denominator))
                    .map_err(|_| "coefficient_cap_division")?;
                correction_sum = correction_sum.and_then(|v| v.checked_add(correction));
                let detail = Remainder {
                    axis,
                    input: query,
                    fit_maximum: maxima[axis],
                    scaled_remainder: remainder,
                    scaled_correction: correction,
                    cap: *cap,
                };
                // Fixed capacity; replacement never transiently adds a fifth item.
                if top.len() < MAX_REMAINDER_DETAILS {
                    top.push(detail);
                } else if correction > top.last().unwrap().scaled_correction {
                    *top.last_mut().unwrap() = detail;
                }
                top.sort_unstable_by(|a, b| {
                    b.scaled_correction
                        .cmp(&a.scaled_correction)
                        .then_with(|| a.axis.cmp(&b.axis))
                });
                bound = bound
                    .checked_add(
                        i128::try_from(correction).map_err(|_| "correction_out_of_i128_range")?,
                    )
                    .ok_or("corrected_bound_overflow")?;
            }
            failure_axis = None;
            result["total_bound_scaled"] = json!(bound.to_string());
            let numerator = u128::try_from(bound).map_err(|_| "negative_final_bound")?;
            let rounded = certificates::ceil_div(numerator, WEIGHT_DENOMINATOR as u128)
                .map_err(|_| "final_division")?;
            u64::try_from(rounded).map_err(|_| "final_bound_out_of_u64_range")
        })();
        result["positive_remainder_axes"] = json!(positive_axes);
        result["positive_remainder_sum_scaled"] = json!(correction_sum.map(|v| v.to_string()));
        result["positive_remainder_ns_ceiling"] = json!(correction_sum
            .and_then(|v| certificates::ceil_div(v, WEIGHT_DENOMINATOR as u128).ok())
            .map(|v| v.to_string()));
        result["largest_positive_remainders"] =
            json!(top.iter().map(Remainder::value).collect::<Vec<_>>());
        result["omitted_positive_remainder_axes"] = json!(positive_axes.saturating_sub(top.len()));
        result["failure_axis"] = json!(failure_axis);
        result["failure_anchor"] = json!(failure_anchor);
        match calculated {
            Ok(upper) => result["upper_ns"] = json!(upper),
            Err(reason) => result["failure"] = json!(reason),
        }
        // Compare with the unchanged production candidate, including None.
        result["matches_original_candidate"] =
            json!(calculated.ok() == self.upper_bound(input, maxima, epsilon_ns, frozen));
        result
    }
}

#[test]
fn signed_bound_diagnostic_preserves_negative_anchor_and_rational_correction() {
    let cache = SignedBasis {
        anchors: vec![Anchor {
            axes: vec![1, 3],
            wall_ns: 10,
        }],
        caps: vec![
            AxisCap {
                numerator: 12,
                denominator: 1,
            },
            AxisCap {
                numerator: 12,
                denominator: 3,
            },
        ],
    };
    let frozen = SignedBasisCertificateV1 {
        anchor_indices: vec![0],
        normalized_query_to_anchor_bits: vec![(-1.0_f64).to_bits(), 0.0_f64.to_bits()],
    };
    let d = cache.diagnose_bound(&[1, 4], &[1, 3], 2, &frozen);
    assert_eq!(d["matches_original_candidate"], true);
    assert_eq!(d["upper_ns"], 44);
    assert_eq!(d["weight_l1"], 1.0);
    assert_eq!(
        d["anchor_bound_scaled"],
        (-8 * WEIGHT_DENOMINATOR).to_string()
    );
    assert_eq!(d["epsilon_allowance_ns_ceiling"], "2");
    assert_eq!(d["positive_remainder_ns_ceiling"], "52");
    assert!(d["failure"].is_null());
}

#[test]
fn signed_bound_diagnostic_reports_unusable_hint_and_caps_detail_count() {
    let cache = SignedBasis {
        anchors: vec![Anchor {
            axes: vec![1; 10],
            wall_ns: 10,
        }],
        caps: vec![
            AxisCap {
                numerator: 10,
                denominator: 1
            };
            10
        ],
    };
    let mut frozen = SignedBasisCertificateV1 {
        anchor_indices: vec![0],
        normalized_query_to_anchor_bits: vec![f64::MAX.to_bits(); 10],
    };
    let failed = cache.diagnose_bound(&[1; 10], &[1; 10], 0, &frozen);
    assert_eq!(failed["failure"], "nonfinite_weight");
    assert_eq!(failed["matches_original_candidate"], true);
    frozen
        .normalized_query_to_anchor_bits
        .fill(0.0_f64.to_bits());
    let bounded = cache.diagnose_bound(&[1; 10], &[1; 10], 0, &frozen);
    assert_eq!(bounded["upper_ns"], 100);
    assert_eq!(bounded["matches_original_candidate"], true);
    assert_eq!(
        bounded["largest_positive_remainders"]
            .as_array()
            .unwrap()
            .len(),
        MAX_REMAINDER_DETAILS
    );
    assert_eq!(bounded["omitted_positive_remainder_axes"], 6);
}

#[test]
fn signed_bound_diagnostic_winner_matches_original_identified_prediction() {
    let anchors = [vec![1, 0, 0], vec![1, 2, 0], vec![1, 0, 4]];
    let axes: Vec<_> = anchors.iter().cycle().take(24).cloned().collect();
    let samples: Vec<_> = axes
        .iter()
        .map(|axes| FitSample {
            wall_ns: 1000 + 100 * axes[1] + 50 * axes[2],
            axes,
        })
        .collect();
    let fit = NonNegativeFit::fit_identified(
        &samples,
        &StructuredSettingsV2 {
            max_wave_ns: 1_000_000,
            ..StructuredSettingsV2::default()
        },
        EnvelopeSettings::default(),
    )
    .unwrap();
    let certificate = fit.certificate().clone();
    for query in [[1, 2, 4], [1, 3, 8], [1, 0, 0]] {
        let original = fit.predict_identified_envelope_detailed(&query).unwrap();
        let d = fit.diagnose_identified_bound(&query);
        assert_eq!(d["selected_upper_ns"], original.upper_ns);
        assert_eq!(d["signed"]["matches_original_candidate"], true);
        assert_eq!(d["rank"], 3);
        assert_eq!(d["epsilon_ns"], certificate.epsilon_ns);
        assert!(d["positive_certificate_count"].as_u64().unwrap() <= MAX_CERTIFICATES as u64);
    }
    assert_eq!(fit.certificate(), &certificate);
}
