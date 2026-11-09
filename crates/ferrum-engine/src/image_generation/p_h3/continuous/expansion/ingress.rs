use super::*;

/// The generation metadata hash is separate from the physical ingress owner. Equal
/// PNG bytes in a different allocation cannot inherit an admitted phase's
/// backing assumptions or replace the image between visual and causal work.
pub(super) struct IngressIdentity {
    fingerprint: [u8; 32],
    image: Option<ManagedHostBytes>,
}
impl IngressIdentity {
    pub fn new(input: ImageRequestRef<'_>) -> Self {
        Self {
            fingerprint: generation_fingerprint(input),
            image: match input {
                ImageRequestRef::Generate(_) => None,
                ImageRequestRef::Edit(request) => Some(request.image.clone()),
            },
        }
    }
    pub fn matches(&self, input: ImageRequestRef<'_>) -> bool {
        self.fingerprint == generation_fingerprint(input)
            && match (&self.image, input) {
                (None, ImageRequestRef::Generate(_)) => true,
                (Some(image), ImageRequestRef::Edit(request)) => image.same_region(&request.image),
                _ => false,
            }
    }
}

fn generation_fingerprint(input: ImageRequestRef<'_>) -> [u8; 32] {
    let generation = match input {
        ImageRequestRef::Generate(request) => request,
        ImageRequestRef::Edit(request) => &request.generation,
    };
    // same_region proves immutable PNG identity without hashing megabytes of
    // encoded image again on every causal token-wave advancement.
    initialize::request_identity(ImageRequestRef::Generate(generation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_generation::ImageEditRequest;

    #[test]
    fn expansion_ingress_requires_original_region_not_equal_bytes_or_another_slice() {
        let host = HostMemoryAccount::new(
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
        );
        let backing = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 32)
            .unwrap()
            .freeze();
        let original = ImageEditRequest {
            generation: ImageGenerationRequest {
                prompt: "make the sky blue".into(),
                width: 512,
                height: 768,
                seed: 2,
                n: 1,
            },
            image: backing.slice(4..8).unwrap(),
        };
        let identity = IngressIdentity::new(ImageRequestRef::Edit(&original));
        assert!(identity.matches(ImageRequestRef::Edit(&original.clone())));
        for image in [
            backing.slice(8..12).unwrap(),
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
                .unwrap()
                .freeze(),
        ] {
            let replaced = ImageEditRequest {
                image,
                ..original.clone()
            };
            assert_eq!(
                initialize::request_identity(ImageRequestRef::Edit(&replaced)),
                initialize::request_identity(ImageRequestRef::Edit(&original))
            );
            assert!(!identity.matches(ImageRequestRef::Edit(&replaced)));
        }
        let mut changed = original.clone();
        changed.generation.seed += 1;
        assert!(!identity.matches(ImageRequestRef::Edit(&changed)));
        assert!(!identity.matches(ImageRequestRef::Generate(&original.generation)));
        drop((changed, original, backing));
        assert_eq!(host.usage().unwrap().allocated_bytes(), 32);
        drop(identity);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    }
}
