use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend(kind);
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    if data.len() % 2 != 0 {
        out.push(0);
    }
}
fn wave(tag: u16, bits: u16, channels: u16, data: &[u8]) -> Vec<u8> {
    let rate = 32000u32;
    let block = channels * (bits / 8);
    let mut format = Vec::new();
    format.extend(tag.to_le_bytes());
    format.extend(channels.to_le_bytes());
    format.extend(rate.to_le_bytes());
    format.extend((rate * u32::from(block)).to_le_bytes());
    format.extend(block.to_le_bytes());
    format.extend(bits.to_le_bytes());
    let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
    chunk(&mut out, b"JUNK", &[1, 2, 3]);
    chunk(&mut out, b"fmt ", &format);
    chunk(&mut out, b"data", data);
    let length = out.len() as u32 - 8;
    out[4..8].copy_from_slice(&length.to_le_bytes());
    out
}
fn admitted(host: &HostMemoryAccount, bytes: &[u8]) -> ManagedHostBytes {
    let mut out = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    out.as_mut().copy_from_slice(bytes);
    out.freeze()
}
fn parse(host: &HostMemoryAccount, bytes: &[u8], decoded: usize) -> Result<WaveAudioPlan> {
    WaveAudioPlan::parse(
        admitted(host, bytes),
        AudioInputLimits {
            maximum_encoded_bytes: 4096,
            maximum_decoded_bytes: decoded,
        },
    )
}
fn values(x: &ManagedHostTensor) -> Vec<f32> {
    x.bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn pcm_mono_expands_exactly_and_stereo_preserves_channel_order() {
    let host = host(4096);
    let data: Vec<u8> = [i16::MIN, 0, i16::MAX, 1]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect();
    let mono = parse(&host, &wave(1, 16, 1, &data), 24).unwrap();
    assert_eq!(
        (
            mono.sampling_rate(),
            mono.source_channels(),
            mono.source_samples_per_channel()
        ),
        (32000, 1, 4)
    );
    let output = mono.prepare_stereo(3, &host).unwrap();
    assert_eq!(output.tensor().dimensions(), [2, 1, 3]);
    assert_eq!(
        values(&output),
        [-1., 0., 32767. / 32768., -1., 0., 32767. / 32768.]
    );
    assert_eq!(
        mono.stereo_backing_bytes(3, &host).unwrap(),
        host.backing_bytes(24).unwrap()
    );
    assert!(mono.prepare_stereo(4, &host).is_err());
    drop(output);
    drop(mono);
    let stereo = parse(&host, &wave(1, 16, 2, &data), 16).unwrap();
    let output = stereo.prepare_stereo(2, &host).unwrap();
    assert_eq!(values(&output), [-1., 32767. / 32768., 0., 1. / 32768.]);
    drop(output);
    drop(stereo);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn pcm_integer_widths_and_float32_keep_original_amplitude_boundaries() {
    let host = host(4096);
    for (bits, data, expected) in [
        (8, vec![0, 128, 255], vec![-1., 0., 127. / 128.]),
        (
            24,
            vec![0, 0, 0x80, 0, 0, 0, 0xff, 0xff, 0x7f],
            vec![-1., 0., 8_388_607. / 8_388_608.],
        ),
        (
            32,
            [i32::MIN, 0, i32::MAX]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            vec![-1., 0., 1.],
        ),
    ] {
        let plan = parse(&host, &wave(1, bits, 1, &data), 24).unwrap();
        let output = plan.prepare_stereo(3, &host).unwrap();
        assert_eq!(&values(&output)[..3], &expected);
    }
    let data: Vec<u8> = [-0.0f32, 1.25, -2.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let plan = parse(&host, &wave(3, 32, 1, &data), 24).unwrap();
    let output = plan.prepare_stereo(3, &host).unwrap();
    assert_eq!(
        values(&output)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [-0.0f32, 1.25, -2.5, -0.0, 1.25, -2.5].map(f32::to_bits)
    );
}
#[test]
fn malformed_container_and_capacity_fail_before_retaining_decoded_memory() {
    let host = host(4096);
    let valid = wave(1, 16, 1, &[0, 0, 1, 0]);
    for mutation in 0..6 {
        let mut b = valid.clone();
        match mutation {
            0 => b.push(0),
            1 => {
                b.pop();
            }
            2 => b[16..20].copy_from_slice(&u32::MAX.to_le_bytes()),
            3 => b[32..34].copy_from_slice(&2u16.to_le_bytes()), // unsupported sample encoding
            4 => b[44..46].copy_from_slice(&7u16.to_le_bytes()), // invalid frame alignment
            5 => b[24..28].copy_from_slice(b"data"),             // duplicate data, no format
            _ => unreachable!(),
        }
        assert!(parse(&host, &b, 16).is_err(), "mutation={mutation}");
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let plan = parse(&host, &valid, 15).unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(plan.prepare_stereo(2, &host).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop(plan);
    let mut data = 1f32.to_le_bytes().to_vec();
    data.extend(f32::NAN.to_le_bytes());
    let plan = parse(&host, &wave(3, 32, 1, &data), 16).unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(plan.prepare_stereo(2, &host).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    assert_eq!(values(&plan.prepare_stereo(1, &host).unwrap()), [1., 1.]);
    let foreign = super::tests::host(4096);
    assert!(plan.prepare_stereo(1, &foreign).is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}
