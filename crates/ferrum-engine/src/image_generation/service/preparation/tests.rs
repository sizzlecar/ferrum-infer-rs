use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "blue cat".into(),
        width: 512,
        height: 1024,
        seed,
        n: 1,
    }
}

#[test]
fn generation_identity_checks_all_inputs_and_exact_batch_order() {
    let marker = Arc::new(());
    let original = request(19);
    let next = request(20);
    let expected = vec![
        GenerationIdentity::from(&original),
        GenerationIdentity::from(&next),
    ];
    let bound = BoundPreparation::new(&marker, expected.clone(), 42);
    bound.validate(&marker, &expected).unwrap();
    assert!(bound.validate(&marker, &expected[..1]).is_err());
    assert!(bound
        .validate(&marker, &[expected[1].clone(), expected[0].clone()])
        .is_err());
    let mut changed = Vec::new();
    let mut prompt = original.clone();
    prompt.prompt = "cyan cat".into(); // Same byte length, distinct content.
    changed.push(prompt);
    let mut width = original.clone();
    width.width *= 2;
    changed.push(width);
    let mut height = original.clone();
    height.height /= 2;
    changed.push(height);
    let mut seed = original.clone();
    seed.seed ^= 1 << 40;
    changed.push(seed);
    let mut n = original.clone();
    n.n = 2;
    changed.push(n);
    for request in changed {
        assert!(bound
            .validate(
                &marker,
                &[GenerationIdentity::from(&request), expected[1].clone()]
            )
            .is_err());
    }
    assert_eq!(bound.into_payload(&marker, &expected).unwrap(), 42);
}

struct PayloadDrop(Arc<AtomicBool>);
impl Drop for PayloadDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[test]
fn foreign_pipeline_token_is_rejected_and_its_unused_payload_released() {
    let original = Arc::new(());
    let foreign = Arc::new(());
    let identities = vec![GenerationIdentity::from(&request(1))];
    let released = Arc::new(AtomicBool::new(false));
    let token = ImageBatchPreparation::new(BoundPreparation::new(
        &original,
        identities.clone(),
        PayloadDrop(released.clone()),
    ));
    let owner = token
        .into_inner::<BoundPreparation<PayloadDrop, GenerationIdentity>>()
        .unwrap();
    assert!(owner.into_payload(&foreign, &identities).is_err());
    assert!(released.load(Ordering::Acquire));
    // Keeping an owner alive preserves its original instance marker even when
    // the pipeline itself moves; this is not a borrowed address comparison.
    let moved = original;
    let bound = BoundPreparation::new(&moved, identities.clone(), ());
    bound.into_payload(&moved, &identities).unwrap();
}

fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            64,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn image(host: &HostMemoryAccount) -> ManagedHostBytes {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
        .unwrap();
    bytes.as_mut().copy_from_slice(&[1, 2, 3, 4]);
    bytes.freeze()
}

#[test]
fn edit_identity_requires_the_original_immutable_source_region() {
    let host = host();
    let marker = Arc::new(());
    let request = ImageEditRequest {
        generation: request(11),
        image: image(&host),
    };
    let identities = vec![EditIdentity::from(&request)];
    let owner = BoundPreparation::new(&marker, identities.clone(), ());
    assert_eq!(host.usage().unwrap().committed_bytes(), 4);
    owner
        .validate(&marker, &[EditIdentity::from(&request.clone())])
        .unwrap();
    let replacement = ImageEditRequest {
        generation: request.generation.clone(),
        image: image(&host),
    };
    assert_eq!(request.image.bytes(), replacement.image.bytes());
    assert!(owner
        .validate(&marker, &[EditIdentity::from(&replacement)])
        .is_err());
    let sliced = ImageEditRequest {
        generation: request.generation.clone(),
        image: request.image.slice(1..4).unwrap(),
    };
    assert!(owner
        .validate(&marker, &[EditIdentity::from(&sliced)])
        .is_err());
    drop((replacement, sliced, request));
    assert_eq!(host.usage().unwrap().committed_bytes(), 4);
    drop((identities, owner));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn generation_preparation_cannot_be_reinterpreted_as_an_edit_preparation() {
    let marker = Arc::new(());
    let token = ImageBatchPreparation::new(BoundPreparation::new(
        &marker,
        vec![GenerationIdentity::from(&request(1))],
        (),
    ));
    assert!(token
        .into_inner::<BoundPreparation<(), EditIdentity>>()
        .is_err());
}
