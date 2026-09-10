//! Shared deterministic CT INT4 bytes and CPU oracle for the raw and vNext launchers.
use half::f16;

#[derive(Clone, Copy)]
pub(crate) struct Fixture {
    pub(crate) name: &'static str,
    pub(crate) rows: usize,
    pub(crate) input_features: usize,
    pub(crate) output_features: usize,
}

pub(crate) const GROUP_SIZE: usize = 32;

pub(crate) const FIXTURES: [Fixture; 4] = [
    Fixture {
        name: "asymmetric-packing",
        rows: 3,
        input_features: 128,
        output_features: 64,
    },
    Fixture {
        name: "single-projection",
        rows: 1,
        input_features: 256,
        output_features: 128,
    },
    Fixture {
        name: "fused-qkv-gate-up",
        rows: 2,
        input_features: 256,
        output_features: 256,
    },
    Fixture {
        name: "mixed-dense-linear-attention-segment",
        rows: 4,
        input_features: 128,
        output_features: 192,
    },
];

pub(crate) struct HostFixture {
    pub(crate) input: Vec<f16>,
    pub(crate) qweight_gptq: Vec<i32>,
    pub(crate) scales_grouped: Vec<f16>,
    pub(crate) zero_points_compressed_tensors: Vec<i32>,
    logical_weights: Vec<u8>,
    logical_scales: Vec<f16>,
    logical_zero_points: Vec<u8>,
}

pub(crate) fn build_host_fixture(fixture: Fixture) -> HostFixture {
    let m = fixture.rows;
    let k = fixture.input_features;
    let n = fixture.output_features;
    let groups = k / GROUP_SIZE;
    assert!(k.is_multiple_of(128));
    assert!(n.is_multiple_of(64));

    let input = (0..m * k)
        .map(|index| {
            let centered = ((index * 29 + m * 17) % 257) as f32 - 128.0;
            f16::from_f32(centered / 113.0)
        })
        .collect::<Vec<_>>();
    let logical_weights = (0..k * n)
        .map(|index| ((index * 7 + index / 11 + n / 64) & 0x0f) as u8)
        .collect::<Vec<_>>();
    let logical_scales = (0..groups * n)
        .map(|index| f16::from_f32(0.0125 + ((index * 13 % 17) as f32) * 0.00075))
        .collect::<Vec<_>>();
    let logical_zero_points = (0..groups * n)
        .map(|index| ((index * 5 + index / 9 + 3) & 0x0f) as u8)
        .collect::<Vec<_>>();

    // Marlin weight repacking consumes GPTQ's `[K / 8, N]` word layout.
    // The production source adapter reaches this layout by transposing the
    // checkpoint's compressed-tensors `[N, K / 8]` storage.
    let mut qweight_gptq = vec![0_i32; (k / 8) * n];
    for packed_input in 0..k / 8 {
        for output in 0..n {
            qweight_gptq[packed_input * n + output] = (0..8).fold(0_u32, |word, lane| {
                word | (u32::from(logical_weights[(packed_input * 8 + lane) * n + output])
                    << (lane * 4))
            }) as i32;
        }
    }

    // compressed-tensors packs zero points as `[N / 8, K / G]` with
    // little-endian output-channel nibbles.
    let mut zero_points_compressed_tensors = vec![0_i32; (n / 8) * groups];
    for packed_output in 0..n / 8 {
        for group in 0..groups {
            zero_points_compressed_tensors[packed_output * groups + group] =
                (0..8).fold(0_u32, |word, lane| {
                    word | (u32::from(logical_zero_points[group * n + packed_output * 8 + lane])
                        << (lane * 4))
                }) as i32;
        }
    }

    HostFixture {
        input,
        qweight_gptq,
        scales_grouped: logical_scales.clone(),
        zero_points_compressed_tensors,
        logical_weights,
        logical_scales,
        logical_zero_points,
    }
}

pub(crate) fn cpu_reference(fixture: Fixture, host: &HostFixture) -> Vec<f16> {
    let m = fixture.rows;
    let k = fixture.input_features;
    let n = fixture.output_features;
    let mut output = vec![f16::ZERO; m * n];
    for row in 0..m {
        for output_feature in 0..n {
            let mut sum = 0.0_f32;
            for input_feature in 0..k {
                let group = input_feature / GROUP_SIZE;
                let quantized = host.logical_weights[input_feature * n + output_feature] as i32;
                let zero_point = host.logical_zero_points[group * n + output_feature] as i32;
                let scale = host.logical_scales[group * n + output_feature].to_f32();
                sum += host.input[row * k + input_feature].to_f32()
                    * (quantized - zero_point) as f32
                    * scale;
            }
            output[row * n + output_feature] = f16::from_f32(sum);
        }
    }
    output
}
