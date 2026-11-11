use super::*;
pub(in crate::image_generation::p_h3::expansion) mod fixture;
use fixture::*;

#[test]
fn frozen_visual_assets_keep_the_original_mapping_across_geometry_and_path_changes() {
    let f = MergedExpansionFixture::new();
    let assets = f.assets();
    let original = f.expansion.model.shared_archive();
    assert!(std::ptr::eq(
        assets.weights().source().archive(),
        original.as_ref()
    ));
    assert!(Arc::ptr_eq(
        &original,
        &f.expansion.visual_model.as_ref().unwrap().shared_archive()
    ));
    assert_eq!(
        assets.profile().as_str(),
        Qwen3VLVisionHfFamily::REFERENCE_PROFILE_ID
    );
    assert_eq!(assets.minimal_family().program().outputs().len(), 1);
    assert_eq!(assets.architecture().output_size(), 2);
    let first = assets.plan(3, 2).unwrap();
    let second = assets.plan(2, 3).unwrap();
    assert_ne!(first.config().grid(), second.config().grid());
    let before = assets.fingerprint().to_owned();
    let path = f.directory.join("model.safetensors");
    let replacement = f.directory.join("new-unused.safetensors");
    std::fs::write(&replacement, b"not the original checkpoint").unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    assert!(Arc::new(
        f.expansion
            .defined
            .prepare(
                &ferrum_models::vnext::qwen35::F16_NUMERICAL_PROFILE_ID
                    .parse()
                    .unwrap()
            )
            .unwrap()
    )
    .managed_safetensors()
    .is_err());
    let _third = assets.plan(3, 2).unwrap();
    assert_eq!(assets.fingerprint(), before);
    let c = assets
        .weights()
        .source()
        .archive()
        .tensor("model.visual.patch_embed.proj.weight")
        .unwrap();
    assert_eq!(c.shape(), [8, 3, 2, 2, 2]);
    assert_eq!(c.dtype(), safetensors::Dtype::BF16);
    assert_eq!(
        assets.source_identity().processor_sha256,
        assets.processor().source_sha256()
    );
}

#[test]
fn visual_profile_and_original_processor_identity_are_sealed_before_runtime_admission() {
    let f = MergedExpansionFixture::new();
    let first = f.assets();
    let cuda =
        ExpansionVisualAssets::from_processor(&f.expansion.model, processor(), limits()).unwrap();
    assert_eq!(
        cuda.profile().as_str(),
        Qwen3VLVisionHfFamily::CUDA_PROFILE_ID
    );
    assert_ne!(first.fingerprint(), cuda.fingerprint());
    assert_eq!(first.source_identity(), cuda.source_identity());
    assert!(ExpansionVisualAssets::from_processor_with_profile(
        &f.expansion.model,
        processor(),
        limits(),
        NumericalProfileId::new("unsupported.visual.profile").unwrap()
    )
    .is_err());
    assert_eq!(first.stored_bytes().unwrap(), cuda.stored_bytes().unwrap());
}

#[test]
fn merged_fixture_attaches_original_sources_with_cuda_profiles_and_bounded_edit_context(
) -> anyhow::Result<()> {
    use crate::image_generation::p_h3::{
        expansion::tests::fixture::EOS_ID, tests::fixture::CompleteFixture,
    };
    use ferrum_models::vnext::qwen35::{
        F16_NUMERICAL_PROFILE_ID, F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID,
    };
    use ferrum_runtime::{
        CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID, ROW_SELECT_F16_OPERATION_ID,
    };

    let f = crate::image_generation::p_h3::expansion::MergedExpansionFixture::new();
    // Artifact roles were already frozen. A neighboring file must not be
    // discovered by reopening the directory during the product attach.
    std::fs::write(
        f.directory.join("unselected.safetensors"),
        b"not a selected artifact",
    )?;
    let complete = CompleteFixture::new()?;
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CUDA(0);
    engine.runtime.max_model_len = Some(64);
    let options = MergedExpansionFixture::native_options(engine);
    let assets = f.attach_cuda(complete.assets()?, options.clone())?;
    assert!(assets.has_prompt_expansion_visual());
    let native = &assets.expansion.as_ref().unwrap().assets;
    assert!(Arc::ptr_eq(
        f.expansion.model.prepared().sources(),
        native.model.prepared().sources(),
    ));
    assert_eq!(
        native.family().numerical_profile().id.as_str(),
        F16_NUMERICAL_PROFILE_ID
    );
    assert_eq!(
        native
            .visual_causal_family()
            .unwrap()
            .numerical_profile()
            .id
            .as_str(),
        F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID
    );
    let visual = native.visual_assets().unwrap();
    assert_eq!(
        visual.profile().as_str(),
        Qwen3VLVisionHfFamily::CUDA_PROFILE_ID
    );
    assert_eq!(
        f.assets().profile().as_str(),
        Qwen3VLVisionHfFamily::REFERENCE_PROFILE_ID
    );
    assert_eq!(
        visual.processor().source_sha256(),
        f.assets().processor().source_sha256()
    );
    let mapping = native.model.shared_archive();
    assert!(Arc::ptr_eq(
        &mapping,
        &native.visual_model.as_ref().unwrap().shared_archive()
    ));
    assert!(std::ptr::eq(
        mapping.as_ref(),
        visual.weights().source().archive()
    ));
    for operation in [
        ROW_SELECT_F16_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID,
    ] {
        assert!(native
            .visual_causal_family()
            .unwrap()
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .any(|node| node.operation_id.as_str() == operation));
    }
    let plan = visual.plan(32, 32)?;
    let grid = plan.config().grid();
    let prepared = native.prepare_edit(
        "cat",
        (32, 32),
        grid,
        visual.architecture().merge_size(),
        &options,
    )?;
    assert_eq!(prepared.generation.eos_token_id, EOS_ID);
    assert_eq!(prepared.generation.max_new_tokens, 2);
    assert_eq!(
        prepared.engine.runtime.max_model_len,
        Some(prepared.tokens.len() + 2)
    );
    assert!(prepared.tokens.len() + 2 <= 64);
    assert_eq!(
        prepared
            .visual
            .as_ref()
            .unwrap()
            .tokens()
            .image_runs()
            .len(),
        1
    );
    assert_eq!(
        prepared.visual.as_ref().unwrap().tokens().image_runs()[0].rows(),
        usize::try_from(grid.rows()? / visual.architecture().merge_size().pow(2))?
    );
    Ok(())
}
