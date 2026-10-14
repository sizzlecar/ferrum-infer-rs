//! Existing M7 MMQ versus M8 MMVQ numerical contracts. This diagnostic does not
//! claim that a batch-width change caused any particular model text divergence.
use super::*;

#[derive(Default)]
struct ErrorStats {
    count: usize,
    sum_squared: f64,
    maximum_absolute: f64,
}

impl ErrorStats {
    fn push(&mut self, error: f64) {
        assert!(error.is_finite());
        self.count += 1;
        self.sum_squared += error * error;
        self.maximum_absolute = self.maximum_absolute.max(error.abs());
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "count":self.count,
            "max_abs":self.maximum_absolute,
            "rms":(self.sum_squared / self.count.max(1) as f64).sqrt(),
        })
    }
}

fn block_bytes(format: UpstreamLinearFormat) -> usize {
    match format {
        UpstreamLinearFormat::Q4K => 144,
        UpstreamLinearFormat::Q5K => 176,
        UpstreamLinearFormat::Iq4Xs => 136,
    }
}

fn distinct_weights(format: UpstreamLinearFormat, k: usize, n: usize) -> Vec<u8> {
    let mut data = weights(format, k, n);
    for (index, block) in data.chunks_exact_mut(block_bytes(format)).enumerate() {
        block[..2].copy_from_slice(
            &f16::from_f32(0.0017 + (index % 11) as f32 * 0.00013)
                .to_bits()
                .to_le_bytes(),
        );
        let start = if format == UpstreamLinearFormat::Iq4Xs {
            2
        } else {
            block[2..4].copy_from_slice(
                &f16::from_f32(0.00031 + (index % 7) as f32 * 0.000037)
                    .to_bits()
                    .to_le_bytes(),
            );
            4
        };
        for (i, byte) in block.iter_mut().enumerate().skip(start) {
            *byte = ((index * 29 + i * 53 + index / 7) % 256) as u8;
        }
    }
    data
}

fn input(k: usize, seed: usize) -> Vec<u16> {
    (0..8 * k)
        .map(|index| {
            let (row, col) = (index / k, index % k);
            if row == 0 && col < 32 {
                return 0x8000; // Exact zero marker, including negative zero.
            }
            if col % 131 == 0 {
                return if row % 2 == 0 { 1 } else { 0x8001 };
            }
            let value = ((col * 37 + row * 19 + seed * 23) % 257) as f32 - 128.0;
            f16::from_f32(value / 97.0 * (1 + (col / 32 + row) % 7) as f32 / 16.0).to_bits()
        })
        .collect()
}

struct PackedGroup {
    scale: f32,
    sum: f32,
    codes: [i8; 32],
}

fn packed_group(route: &Route, packed: &[u8], row: usize, group: usize) -> PackedGroup {
    let p = &route.plan;
    let (metadata, codes) = if route.algorithm == UpstreamLinearAlgorithm::Mmq {
        let block = (group / 4) * p.request.rows as usize + row;
        (
            &packed[block * 144 + (group % 4) * 4..][..4],
            &packed[block * 144 + 16 + (group % 4) * 32..][..32],
        )
    } else {
        let block = row * (p.padded_inputs as usize / 32) + group;
        (&packed[block * 36..][..4], &packed[block * 36 + 4..][..32])
    };
    PackedGroup {
        scale: if p.pack_abi == 1 {
            f32::from_le_bytes(metadata.try_into().unwrap())
        } else {
            f16::from_bits(u16::from_le_bytes(metadata[..2].try_into().unwrap())).to_f32()
        },
        sum: if p.pack_abi == 1 {
            0.0
        } else {
            f16::from_bits(u16::from_le_bytes(metadata[2..].try_into().unwrap())).to_f32()
        },
        codes: std::array::from_fn(|i| codes[i] as i8),
    }
}

struct Observation {
    first_seven_bits: Vec<u16>,
    first_seven_f32: Vec<f32>,
    report: serde_json::Value,
}

fn observe(
    route: &Route,
    format: UpstreamLinearFormat,
    weights: &[u8],
    input: &[u16],
) -> Observation {
    let p = &route.plan;
    let (m, k, n) = (
        p.request.rows as usize,
        p.request.inputs as usize,
        p.request.outputs as usize,
    );
    // Existing route checker validates flags, final cast, full output-window
    // canaries and allocation guards. Below extends its F64 oracle to every
    // output of the seven common rows, not merely representative coordinates.
    let bits = route.validate(format, weights, 1);
    let values = route.output.read::<f32>(m * n);
    let packed = route.packed.read::<u8>(p.packed_bytes as usize);
    if route.algorithm == UpstreamLinearAlgorithm::Mmq {
        let body = m * p.padded_inputs as usize * 9 / 8;
        assert!(packed[body..].iter().all(|byte| *byte == 0));
    }
    let mut implementation = ErrorStats::default();
    let mut input_quantization = ErrorStats::default();
    let mut declared_correction = ErrorStats::default();
    let mut f32_total = ErrorStats::default();
    let mut f16_total = ErrorStats::default();
    let mut cast = ErrorStats::default();
    let mut activation_quantization = ErrorStats::default();
    let mut maximum_bound_ratio = 0.0_f64;
    for row in 0..7 {
        let groups: Vec<_> = (0..k / 32)
            .map(|g| packed_group(route, &packed, row, g))
            .collect();
        for (group, pack) in groups.iter().enumerate() {
            let x = std::array::from_fn(|i| input[row * k + group * 32 + i]);
            let (zero, bad) = oracle::marker_classify_pack(route.algorithm, format, &x);
            assert!(!bad);
            assert!(pack.scale.is_finite() && pack.scale >= 0.0);
            assert!(pack.codes.iter().all(|code| (-127..=127).contains(code)));
            if zero {
                assert_eq!(pack.scale.to_bits(), 0);
                assert_eq!(pack.sum.to_bits(), 0);
                assert_eq!(pack.codes, [0; 32]);
            }
            for (i, code) in pack.codes.iter().enumerate() {
                activation_quantization.push(
                    f64::from(pack.scale) * f64::from(*code)
                        - f64::from(f16::from_bits(x[i]).to_f32()),
                );
            }
        }
        for col in 0..n {
            let (mut declared, mut magnitude, mut decoded_packed, mut common) =
                (0.0, 0.0, 0.0, 0.0);
            for (group, pack) in groups.iter().enumerate() {
                let offset = (col * (k / 256) + group / 8) * block_bytes(format);
                let block = &weights[offset..offset + block_bytes(format)];
                let (target, mag) = oracle::declared_group(
                    route.algorithm,
                    format,
                    block,
                    group % 8,
                    &pack.codes,
                    pack.scale,
                    pack.sum,
                );
                declared += target;
                magnitude += mag;
                let decoded = oracle::decoded_group(format, block, group % 8);
                for i in 0..32 {
                    decoded_packed += decoded[i] * f64::from(pack.scale) * f64::from(pack.codes[i]);
                    common += decoded[i]
                        * f64::from(f16::from_bits(input[row * k + group * 32 + i]).to_f32());
                }
            }
            let actual = f64::from(values[row * n + col]);
            // Same established actual-pack forward-error bound as Route::validate;
            // neither the other GPU route nor raw-input error sets this bound.
            let nu = (k as f64 / 32.0 + 32.0) * f64::from(f32::EPSILON);
            let bound = nu / (1.0 - nu) * magnitude + 32.0 * f64::from(f32::MIN_POSITIVE);
            assert!(
                (actual - declared).abs() <= bound,
                "{:?}/{format:?} row{row} col{col}: actual {actual}, declared {declared}, bound {bound}",
                route.algorithm
            );
            maximum_bound_ratio = maximum_bound_ratio.max((actual - declared).abs() / bound);
            let final_value = f64::from(f16::from_bits(bits[row * n + col]).to_f32());
            implementation.push(actual - declared);
            input_quantization.push(decoded_packed - common);
            declared_correction.push(declared - decoded_packed);
            f32_total.push(actual - common);
            cast.push(final_value - actual);
            f16_total.push(final_value - common);
        }
    }
    Observation {
        first_seven_bits: bits[..7 * n].to_vec(),
        first_seven_f32: values[..7 * n].to_vec(),
        report: serde_json::json!({
            "algorithm":format!("{:?}",route.algorithm),"rows":m,"pack_abi":p.pack_abi,
            "kernel_vs_declared_actual_pack":implementation.json(),
            "maximum_kernel_error_bound_ratio":maximum_bound_ratio,
            "decoded_weight_dot_input_quantization_error":input_quantization.json(),
            "declared_coefficient_and_affine_correction_error":declared_correction.json(),
            "f32_vs_common_decoded_weight_raw_input":f32_total.json(),
            "f16_cast_error":cast.json(),
            "f16_vs_common_decoded_weight_raw_input":f16_total.json(),
            "activation_dequantization_error":activation_quantization.json(),
            "pack_oracle_scope":"actual GPU pack codes/scales + independent marker classification; fast-math ideal CPU pack bits are not asserted",
        }),
    }
}

#[test]
#[ignore = "requires CUDA and linked qualified archive; numerical diagnostic, no serving claim"]
fn m7_mmq_m8_mmvq_report_separate_pack_and_kernel_errors() {
    let _ = caps();
    for (format, k) in [
        (UpstreamLinearFormat::Q4K, 5120),
        (UpstreamLinearFormat::Q5K, 6144),
        (UpstreamLinearFormat::Iq4Xs, 5120),
    ] {
        let n = 64;
        let stream = Stream::new();
        let routes = [
            Route::new_with_result_window(
                UpstreamLinearAlgorithm::Mmq,
                format,
                7,
                k,
                n,
                1,
                n + 5,
                1,
            ),
            Route::new_with_result_window(
                UpstreamLinearAlgorithm::Mmvq,
                format,
                8,
                k,
                n,
                1,
                n + 5,
                1,
            ),
        ];
        let weights = distinct_weights(format, k as usize, n as usize);
        let weight = Buffer::new(weights.len());
        weight.write(&weights);
        let activation = Buffer::new(8 * k as usize * 2);
        let original = input(k as usize, 0);
        activation.write(&original);
        for route in &routes {
            route.scan(&stream, weight.ptr, 0);
        }
        stream.sync();
        let graphs: Vec<_> = routes
            .iter()
            .map(|route| {
                Graph::capture(&stream, || {
                    route.enqueue_with_mmvq_dot(
                        &stream,
                        &activation,
                        weight.ptr,
                        0,
                        shared_weight_decode::selected_dot(format),
                    );
                })
            })
            .collect();
        let mut first_bits: Option<[Vec<u16>; 2]> = None;
        let mut first_f32: Option<[Vec<u32>; 2]> = None;
        for phase in [
            "same_first_seven",
            "eighth_only_changed",
            "fresh_common_rows",
        ] {
            let mut host_input = if phase == "fresh_common_rows" {
                input(k as usize, 1)
            } else {
                original.clone()
            };
            if phase == "eighth_only_changed" {
                for value in &mut host_input[7 * k as usize..] {
                    *value ^= 0x8000;
                }
            }
            activation.write(&host_input);
            for graph in &graphs {
                graph.launch(&stream);
            }
            stream.sync();
            let observations: [Observation; 2] =
                std::array::from_fn(|i| observe(&routes[i], format, &weights, &host_input));
            let bit_pair = observations.each_ref().map(|o| o.first_seven_bits.clone());
            let f32_pair = observations.each_ref().map(|o| {
                o.first_seven_f32
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            });
            if phase == "eighth_only_changed" {
                assert_eq!(first_bits.as_ref().unwrap(), &bit_pair);
                assert_eq!(first_f32.as_ref().unwrap(), &f32_pair);
            } else if phase == "same_first_seven" {
                first_bits = Some(bit_pair.clone());
                first_f32 = Some(f32_pair.clone());
            }
            let differing_f16 = bit_pair[0]
                .iter()
                .zip(&bit_pair[1])
                .filter(|(a, b)| a != b)
                .count();
            let mut route_difference = ErrorStats::default();
            for (a, b) in observations[0]
                .first_seven_f32
                .iter()
                .zip(&observations[1].first_seven_f32)
            {
                route_difference.push(f64::from(*b) - f64::from(*a));
            }
            assert_eq!(activation.read::<u16>(host_input.len()), host_input);
            assert_eq!(weight.read::<u8>(weights.len()), weights);
            activation.guards();
            weight.guards();
            println!(
                "{}",
                serde_json::json!({
                    "kind":"batch_width_numerical_boundary","format":format!("{format:?}"),
                    "k":k,"n":n,"common_rows":7,"phase":phase,
                    "mmq_m7":observations[0].report,"mmvq_m8":observations[1].report,
                    "m8_dot":if format == UpstreamLinearFormat::Iq4Xs {"baseline"} else {"shared_m8"},
                    "cross_route_f16_different_values":differing_f16,
                    "cross_route_f32_difference":route_difference.json(),
                    "scope":"same weights and first seven F16 rows; complete outputs, actual-pack F64 bounds, guards and captured fresh input; no performance or model divergence causality claim",
                })
            );
        }
    }
}
