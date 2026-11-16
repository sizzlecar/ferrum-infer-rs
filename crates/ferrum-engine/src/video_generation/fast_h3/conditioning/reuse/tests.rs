use super::*;
use crate::video_generation::{VideoFrameRate, VideoGenerationParameters, VideoReference};
use ferrum_runtime::*;

fn request(prompt: &str, images: &[&[u8]]) -> VideoGenerationRequest {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    VideoGenerationRequest {
        parameters: VideoGenerationParameters {
            prompt: prompt.into(),
            width: 832,
            height: 480,
            num_frames: 124,
            frame_rate: VideoFrameRate {
                numerator: 24,
                denominator: 1,
            },
            seed: 1101,
        },
        references: images
            .iter()
            .map(|bytes| {
                let mut output = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                    .unwrap();
                output.as_mut().copy_from_slice(bytes);
                VideoReference::Image(output.freeze())
            })
            .collect(),
    }
}

#[test]
fn temporal_conditioning_identity_includes_kind_order_and_trim_duration() {
    let original = request("wave", &[b"visual", b"sound"]);
    let mut audio = original.clone();
    audio.references[1] = VideoReference::Audio(audio.references[1].bytes().clone());
    let key = ConditioningKey::new(&audio);
    assert!(key != ConditioningKey::new(&original));
    let mut reordered = audio.clone();
    reordered.references.swap(0, 1);
    assert!(key != ConditioningKey::new(&reordered));
    let mut duration = audio.clone();
    duration.parameters.num_frames += 17;
    assert!(key != ConditioningKey::new(&duration));
    let mut rate = audio.clone();
    rate.parameters.frame_rate.numerator *= 2;
    assert!(key != ConditioningKey::new(&rate));
    let mut generation = audio.clone();
    generation.parameters.seed += 1;
    generation.parameters.width *= 2;
    assert!(key == ConditioningKey::new(&generation));
}

#[test]
fn conditioning_identity_uses_raw_prompt_and_ordered_content_not_generation_noise() {
    let original = request("wave", &[b"image one", b"image two"]);
    let key = ConditioningKey::new(&original);
    let mut generated = original.clone();
    generated.parameters.seed += 1;
    generated.parameters.width = 1280;
    generated.parameters.height = 768;
    generated.parameters.num_frames = 243;
    assert!(key == ConditioningKey::new(&generated));
    assert!(key == ConditioningKey::new(&request("wave", &[b"image one", b"image two"])));
    for changed in [
        request("wave ", &[b"image one", b"image two"]),
        request("wave", &[b"image One", b"image two"]),
        request("wave", &[b"image two", b"image one"]),
        request("wave", &[b"image oneimage two"]),
    ] {
        assert!(key != ConditioningKey::new(&changed));
    }
    assert!(
        ConditioningKey::new(&request("ab", &[b"c"]))
            != ConditioningKey::new(&request("a", &[b"bc"]))
    );
}

#[test]
fn reuse_capacity_is_explicit_and_includes_the_last_byte() {
    let policy: FastH3ConditioningReuse =
        serde_json::from_str(r#"{"kind":"last_result","maximum_retained_bytes":128}"#).unwrap();
    policy.validate().unwrap();
    assert!(policy.retains(128));
    assert!(!policy.retains(129));
    assert!(!FastH3ConditioningReuse::Disabled {}.retains(1));
    assert!(FastH3ConditioningReuse::LastResult {
        maximum_retained_bytes: 0
    }
    .validate()
    .is_err());
    assert!(serde_json::from_str::<FastH3ConditioningReuse>(
        r#"{"kind":"disabled","unexpected":1}"#
    )
    .is_err());
}
