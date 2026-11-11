use super::*;
use crate::p_h3_image_vae::latent_statistics::tests::{config, source};

fn host(capacity: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            capacity,
            HostMemoryClassLimits {
                input_preprocess_bytes: capacity,
                output_bytes: capacity,
                weight_backing_bytes: 0,
                transfer_staging_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn spec(dimensions: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        dimensions.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn bytes(host: &HostMemoryAccount, class: HostMemoryClass, values: &[f32]) -> ManagedHostBytes {
    let mut allocation = host.allocate_zeroed(class, values.len() * 4).unwrap();
    for (dst, value) in allocation.as_mut().chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    allocation.freeze()
}
fn input(host: &HostMemoryAccount, dimensions: &[u64], values: &[f32]) -> ManagedHostTensor {
    ManagedHostTensor::new(
        id(),
        spec(dimensions),
        bytes(host, HostMemoryClass::InputPreprocess, values),
    )
    .unwrap()
}
fn id() -> ProgramValueId {
    ProgramValueId::new("value.encoder_boundary").unwrap()
}
fn values(t: &ManagedHostTensor) -> Vec<f32> {
    t.bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn pixel_domains_preserve_wrapper_rounding_nchw_and_original_owners() {
    let account = host(4096);
    let unit = [2.0_f32.powi(-26), 1., 0., 1., 0.5, 0.25];
    let signed = unit.map(|v| {
        let scaled = v * 2.;
        scaled - 1.
    });
    let a = input(&account, &[1, 3, 1, 2], &unit);
    let b = input(&account, &[1, 3, 1, 2], &signed);
    let original = a.bytes().bytes().to_vec();
    let normalized =
        normalized_pixels(&account, &a, PH3ImageVaePixelDomain::UnitRgb, id()).unwrap();
    let other = normalized_pixels(&account, &b, PH3ImageVaePixelDomain::SignedRgb, id()).unwrap();
    assert_eq!(normalized.bytes().bytes(), other.bytes().bytes());
    let result = values(&normalized);
    assert_eq!(result[0], -0.485_f32 / 0.229); // round-trip loses the tiny positive pixel
    assert_eq!(result[1], (1.0_f32 - 0.485) / 0.229);
    assert_eq!(result[2], -0.456_f32 / 0.224);
    assert_eq!(result[5], (0.25_f32 - 0.406) / 0.225);
    assert_eq!(a.bytes().bytes(), original);
    let before = account.usage().unwrap().committed_bytes();
    let foreign = host(4096);
    assert!(normalized_pixels(&foreign, &a, PH3ImageVaePixelDomain::UnitRgb, id()).is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    for value in [-0.01, 1.01, f32::NAN] {
        let invalid = input(&account, &[1, 3, 1, 2], &[0., 0., 0., 0., 0., value]);
        let with_input = account.usage().unwrap().committed_bytes();
        assert!(
            normalized_pixels(&account, &invalid, PH3ImageVaePixelDomain::UnitRgb, id()).is_err()
        );
        assert_eq!(account.usage().unwrap().committed_bytes(), with_input);
    }
    assert_eq!(account.usage().unwrap().committed_bytes(), before);
    drop((a, b, normalized, other));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn posterior_splits_mean_logvar_per_request_and_preserves_b1_equivalence() {
    let account = host(8192);
    let (_dir, original, config) = {
        let (d, s) = source(safetensors::Dtype::F32, &[1., 10.], &[2., 4.]);
        (d, s, config())
    };
    let statistics = PH3ImageVaeLatentStatistics::new(&config, &original).unwrap();
    let moment_values = [
        1., 2., 10., 20., 0., 0., 0., 0., 101., 102., 110., 120., 0., 0., 0., 0.,
    ];
    let noise_values = [3., -3., 4., -4., 13., -13., 14., -14.];
    let moments = bytes(&account, HostMemoryClass::Output, &moment_values);
    let noise = input(&account, &[2, 2, 1, 2], &noise_values);
    let result = posterior_from_noise(
        &account,
        &spec(&[2, 4, 1, 2]),
        &moments,
        &noise,
        &statistics,
        id(),
    )
    .unwrap();
    assert_eq!(values(&result), [1.5, -1., 1., 1.5, 56.5, 44., 28.5, 24.]);
    for b in 0..2 {
        let one_moments = bytes(
            &account,
            HostMemoryClass::Output,
            &moment_values[b * 8..(b + 1) * 8],
        );
        let one_noise = input(&account, &[1, 2, 1, 2], &noise_values[b * 4..(b + 1) * 4]);
        let one = posterior_from_noise(
            &account,
            &spec(&[1, 4, 1, 2]),
            &one_moments,
            &one_noise,
            &statistics,
            id(),
        )
        .unwrap();
        assert_eq!(values(&one), values(&result)[b * 4..(b + 1) * 4]);
    }
    assert_eq!(values(&noise), noise_values);
    assert_eq!(
        moments.bytes(),
        moment_values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    drop((moments, noise, result));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn posterior_clamps_logvar_and_uses_fused_addcmul_before_channel_normalization() {
    let account = host(4096);
    let (_dir, original) = source(safetensors::Dtype::F32, &[0., 0.], &[1., 1.]);
    let config = config();
    let statistics = PH3ImageVaeLatentStatistics::new(&config, &original).unwrap();
    let logarithm = 2.0_f32 * libm::log1pf(f32::EPSILON);
    let scale = libm::expf(logarithm * 0.5);
    assert_eq!(scale, 1.0 + f32::EPSILON);
    let moments = bytes(
        &account,
        HostMemoryClass::Output,
        &[-1.5, 0., logarithm, 0.],
    );
    let noise = input(&account, &[1, 2, 1, 1], &[1.5, 0.]);
    let sample = posterior_from_noise(
        &account,
        &spec(&[1, 4, 1, 1]),
        &moments,
        &noise,
        &statistics,
        id(),
    )
    .unwrap();
    assert_eq!(values(&sample)[0], 3.0 * 2.0_f32.powi(-24));
    assert_ne!(values(&sample)[0], scale * 1.5 - 1.5);
    let extremes = bytes(&account, HostMemoryClass::Output, &[0., 0., -1000., 1000.]);
    let noise = input(&account, &[1, 2, 1, 1], &[1., 1.]);
    let sample = posterior_from_noise(
        &account,
        &spec(&[1, 4, 1, 1]),
        &extremes,
        &noise,
        &statistics,
        id(),
    )
    .unwrap();
    for (actual, exponent) in values(&sample).into_iter().zip([-15.0_f64, 10.]) {
        let oracle = exponent.exp();
        assert!((f64::from(actual) - oracle).abs() <= 1e-7 * oracle);
    }
}

#[test]
fn posterior_late_invalid_or_capacity_failure_keeps_sources_and_refunds_successor() {
    let probe = host(4096);
    let capacity = probe.backing_bytes(16).unwrap() + 2 * probe.backing_bytes(8).unwrap();
    let account = host(capacity);
    let (_dir, original) = source(safetensors::Dtype::F32, &[0., 0.], &[1., 1.]);
    let config = config();
    let statistics = PH3ImageVaeLatentStatistics::new(&config, &original).unwrap();
    let moments = bytes(&account, HostMemoryClass::Output, &[1., 2., 0., 0.]);
    let noise = input(&account, &[1, 2, 1, 1], &[0., 0.]);
    let before = account.usage().unwrap().committed_bytes();
    let blocker = account.allocate_zeroed(HostMemoryClass::Output, 1).unwrap();
    let blocked = account.usage().unwrap().committed_bytes();
    assert!(posterior_from_noise(
        &account,
        &spec(&[1, 4, 1, 1]),
        &moments,
        &noise,
        &statistics,
        id()
    )
    .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), blocked);
    drop(blocker);
    let sample = posterior_from_noise(
        &account,
        &spec(&[1, 4, 1, 1]),
        &moments,
        &noise,
        &statistics,
        id(),
    )
    .unwrap();
    assert_eq!(values(&sample), [1., 2.]);
    drop(sample);
    assert_eq!(account.usage().unwrap().committed_bytes(), before);
    drop(moments);
    let bad = bytes(&account, HostMemoryClass::Output, &[1., 2., 0., f32::NAN]);
    assert!(posterior_from_noise(
        &account,
        &spec(&[1, 4, 1, 1]),
        &bad,
        &noise,
        &statistics,
        id()
    )
    .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), before);
    assert_eq!(values(&noise), [0., 0.]);
    drop((bad, noise));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
