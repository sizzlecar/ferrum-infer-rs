use super::*;
use crate::image_generation::ImageEditRequest;

pub(super) enum EffectiveRequest {
    Generate(ImageGenerationRequest),
    Edit(ImageEditRequest),
}
impl EffectiveRequest {
    pub fn from_expanded(input: ImageRequestRef<'_>, prompt: String) -> Self {
        match input {
            ImageRequestRef::Generate(original) => Self::Generate(ImageGenerationRequest {
                prompt,
                ..original.clone()
            }),
            ImageRequestRef::Edit(original) => Self::Edit(ImageEditRequest {
                generation: ImageGenerationRequest {
                    prompt,
                    ..original.generation.clone()
                },
                image: original.image.clone(),
            }),
        }
    }
    pub fn as_ref(&self) -> ImageRequestRef<'_> {
        match self {
            Self::Generate(request) => ImageRequestRef::Generate(request),
            Self::Edit(request) => ImageRequestRef::Edit(request),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expanded_edit_preserves_exact_ingress_backing_and_generation_controls() {
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
        let image = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
            .unwrap()
            .freeze();
        let original = ImageEditRequest {
            generation: ImageGenerationRequest {
                prompt: "change the sky".into(),
                width: 512,
                height: 768,
                seed: 0x1234_5678_0000_0001,
                n: 1,
            },
            image,
        };
        let ingress = initialize::request_identity(ImageRequestRef::Edit(&original));
        let before = host.usage().unwrap().allocated_bytes();
        let effective = EffectiveRequest::from_expanded(
            ImageRequestRef::Edit(&original),
            "make the sky blue".into(),
        );
        let ImageRequestRef::Edit(edit) = effective.as_ref() else {
            panic!("edit kind changed")
        };
        assert!(edit.image.same_region(&original.image));
        assert_eq!(host.usage().unwrap().allocated_bytes(), before);
        assert_eq!(
            (
                edit.generation.width,
                edit.generation.height,
                edit.generation.seed,
                edit.generation.n
            ),
            (
                original.generation.width,
                original.generation.height,
                original.generation.seed,
                original.generation.n
            )
        );
        assert_eq!(edit.generation.prompt, "make the sky blue");
        assert_eq!(
            initialize::request_identity(ImageRequestRef::Edit(&original)),
            ingress
        );
        assert_ne!(initialize::request_identity(effective.as_ref()), ingress);
        drop(original);
        assert!(host.usage().unwrap().allocated_bytes() > 0);
        drop(effective);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    }
}
