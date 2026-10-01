//! Cold geometry instances of existing product roots. These are not additional
//! host policies or evidence that an actual execution reaches a required route.
use super::geometry::DecodeInterval;
use crate::{AutomaticCostProbePromptBudget, AutomaticCostProbeTemplate};
use ferrum_interfaces::Tokenizer;
use ferrum_types::{FerrumError, Result};
use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroUsize},
};

#[derive(Debug)]
pub(super) struct ContextVariant<'a> {
    pub original_template_index: usize,
    pub template: Cow<'a, AutomaticCostProbeTemplate>,
    pub prompt_tokens: NonZeroUsize,
    pub maximum_output: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ContextVariantGap {
    IntervalOutsideContext,
    NoPostReleaseDecodeWindow,
    NoContinuationPrefillWindow,
    RendererUnavailableOrUnreachableWindow,
    RenderedPromptMissesInterval,
    RenderedPromptMissesContinuationWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContextVariantResolution {
    Covered { variant_index: usize },
    Gap { reason: ContextVariantGap },
}

#[derive(Debug)]
pub(super) struct ContextVariantInventory<'a> {
    context: usize,
    configured_output: NonZeroUsize,
    maximum_inventory_bytes: usize,
    variants: Vec<ContextVariant<'a>>,
}

impl<'a> ContextVariantInventory<'a> {
    /// The caller allocates this allowance from its shared retained capacity.
    /// Borrowed roots retain no additional payload. Rendered payload is already
    /// charged to the caller's shared prompt budget; this limit covers the Vec.
    pub fn new(
        context: usize,
        configured_output: NonZeroUsize,
        maximum_inventory_bytes: usize,
    ) -> Result<Self> {
        if context < 2 || maximum_inventory_bytes < std::mem::size_of::<ContextVariant<'a>>() {
            return Err(FerrumError::invalid_request(
                "probe context inventory has no complete-request capacity",
            ));
        }
        Ok(Self {
            context,
            configured_output,
            maximum_inventory_bytes,
            variants: Vec::new(),
        })
    }

    pub fn variants(&self) -> &[ContextVariant<'a>] {
        &self.variants
    }

    pub fn inventory_retained_bytes(&self) -> usize {
        self.variants.capacity() * std::mem::size_of::<ContextVariant<'a>>()
    }

    /// `original_prompt_tokens` is the caller's one actual tokenizer result for
    /// the unchanged root. Reuse spans intervals and prefix releases; an exact
    /// numerical family still has to be checked by the downstream inventory.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &mut self,
        original_template_index: usize,
        original: &'a AutomaticCostProbeTemplate,
        original_prompt_tokens: NonZeroUsize,
        release_generated: NonZeroUsize,
        interval: DecodeInterval,
        tokenizer: &dyn Tokenizer,
        budget: &mut AutomaticCostProbePromptBudget,
    ) -> Result<ContextVariantResolution> {
        self.resolve_window(
            original_template_index,
            original,
            original_prompt_tokens,
            release_generated,
            interval,
            false,
            tokenizer,
            budget,
        )
    }

    /// A fresh guaranteed member belongs to the first ordinary suffix wave.
    /// Merely reaching a later context in an output window is insufficient:
    /// EOS or pending bytes can change that later wave's population.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_first_suffix(
        &mut self,
        original_template_index: usize,
        original: &'a AutomaticCostProbeTemplate,
        original_prompt_tokens: NonZeroUsize,
        release_generated: NonZeroUsize,
        sequence_tokens: u32,
        tokenizer: &dyn Tokenizer,
        budget: &mut AutomaticCostProbePromptBudget,
    ) -> Result<ContextVariantResolution> {
        self.resolve_window(
            original_template_index,
            original,
            original_prompt_tokens,
            release_generated,
            DecodeInterval {
                first_sequence_tokens: sequence_tokens,
                last_sequence_tokens: sequence_tokens,
            },
            true,
            tokenizer,
            budget,
        )
    }

    /// A cold ordinary request must consume its prompt before sampling can
    /// observe EOS. Three legal spans provide both a complete intermediate
    /// continuation and the final span without requiring a context-sized root.
    /// The checked input inventory still owns all route and freshness proofs.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_continuation_prefill(
        &mut self,
        original_template_index: usize,
        original: &'a AutomaticCostProbeTemplate,
        original_prompt_tokens: NonZeroUsize,
        per_row_chunk: NonZeroUsize,
        tokenizer: &dyn Tokenizer,
        budget: &mut AutomaticCostProbePromptBudget,
    ) -> Result<ContextVariantResolution> {
        self.register_root(original_template_index, original, original_prompt_tokens)?;
        let minimum = per_row_chunk
            .get()
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .and_then(NonZeroUsize::new)
            .ok_or_else(capacity_error)?;
        let maximum = per_row_chunk
            .get()
            .checked_mul(3)
            .ok_or_else(capacity_error)?
            .min(self.context - 1);
        let Some(maximum) = NonZeroUsize::new(maximum).filter(|n| *n >= minimum) else {
            return Ok(ContextVariantResolution::Gap {
                reason: ContextVariantGap::NoContinuationPrefillWindow,
            });
        };
        let matches_window = |prompt: NonZeroUsize| minimum <= prompt && prompt <= maximum;
        if let Some(variant_index) = self.variants.iter().position(|v| {
            v.original_template_index == original_template_index && matches_window(v.prompt_tokens)
        }) {
            return Ok(ContextVariantResolution::Covered { variant_index });
        }
        // This consumes the same allowance as decode geometry. There is no
        // independent renderer budget or uncharged temporary variant storage.
        self.reserve_variant()?;
        let Some(rendered) =
            original.for_prompt_token_window(tokenizer, minimum, maximum, budget)?
        else {
            return Ok(ContextVariantResolution::Gap {
                reason: ContextVariantGap::RendererUnavailableOrUnreachableWindow,
            });
        };
        let Some(maximum_output) = self.output_for(rendered.prompt_tokens) else {
            return Ok(ContextVariantResolution::Gap {
                reason: ContextVariantGap::RenderedPromptMissesContinuationWindow,
            });
        };
        if !matches_window(rendered.prompt_tokens) {
            return Ok(ContextVariantResolution::Gap {
                reason: ContextVariantGap::RenderedPromptMissesContinuationWindow,
            });
        }
        let variant_index = self.variants.len();
        self.variants.push(ContextVariant {
            original_template_index,
            template: Cow::Owned(rendered.template),
            prompt_tokens: rendered.prompt_tokens,
            maximum_output,
        });
        Ok(ContextVariantResolution::Covered { variant_index })
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_window(
        &mut self,
        original_template_index: usize,
        original: &'a AutomaticCostProbeTemplate,
        original_prompt_tokens: NonZeroUsize,
        release_generated: NonZeroUsize,
        interval: DecodeInterval,
        first_suffix: bool,
        tokenizer: &dyn Tokenizer,
        budget: &mut AutomaticCostProbePromptBudget,
    ) -> Result<ContextVariantResolution> {
        if interval.first_sequence_tokens > interval.last_sequence_tokens {
            return Err(FerrumError::invalid_request(
                "reversed probe context interval",
            ));
        }
        let gap = |reason| Ok(ContextVariantResolution::Gap { reason });
        if interval.first_sequence_tokens < 2
            || interval.last_sequence_tokens as usize >= self.context
        {
            return gap(ContextVariantGap::IntervalOutsideContext);
        }
        let Some((minimum, maximum)) =
            interval.prompt_window(release_generated, self.configured_output)
        else {
            return gap(ContextVariantGap::NoPostReleaseDecodeWindow);
        };
        let minimum = if first_suffix { maximum } else { minimum };
        let matches_window = |prompt: NonZeroUsize, output: NonZeroUsize| {
            covers(prompt, output, release_generated, interval)
                && (!first_suffix || prompt == maximum)
        };
        self.register_root(original_template_index, original, original_prompt_tokens)?;
        if let Some(variant_index) = self.variants.iter().position(|v| {
            v.original_template_index == original_template_index
                && matches_window(v.prompt_tokens, v.maximum_output)
        }) {
            return Ok(ContextVariantResolution::Covered { variant_index });
        }
        // Authorize inventory storage before any renderer/tokenizer work. The
        // shared prompt budget separately charges the new template's payload.
        self.reserve_variant()?;
        let Some(rendered) =
            original.for_prompt_token_window(tokenizer, minimum, maximum, budget)?
        else {
            return gap(ContextVariantGap::RendererUnavailableOrUnreachableWindow);
        };
        let Some(maximum_output) = self.output_for(rendered.prompt_tokens) else {
            return gap(ContextVariantGap::RenderedPromptMissesInterval);
        };
        if !matches_window(rendered.prompt_tokens, maximum_output) {
            return gap(ContextVariantGap::RenderedPromptMissesInterval);
        }
        let variant_index = self.variants.len();
        self.variants.push(ContextVariant {
            original_template_index,
            template: Cow::Owned(rendered.template),
            prompt_tokens: rendered.prompt_tokens,
            maximum_output,
        });
        Ok(ContextVariantResolution::Covered { variant_index })
    }

    fn register_root(
        &mut self,
        original_template_index: usize,
        original: &'a AutomaticCostProbeTemplate,
        original_prompt_tokens: NonZeroUsize,
    ) -> Result<()> {
        let original_output = self.output_for(original_prompt_tokens).ok_or_else(|| {
            FerrumError::invalid_request("original probe prompt leaves no output room")
        })?;
        if let Some(root) = self.variants.iter().find(|v| {
            v.original_template_index == original_template_index
                && matches!(v.template, Cow::Borrowed(_))
        }) {
            if root.prompt_tokens != original_prompt_tokens
                || root.template.serialized_request() != original.serialized_request()
                || root.template.output() != original.output()
                || root.template.response_model() != original.response_model()
            {
                return Err(FerrumError::invalid_request(
                    "probe context root index was reused for a different original",
                ));
            }
        } else {
            self.reserve_variant()?;
            self.variants.push(ContextVariant {
                original_template_index,
                template: Cow::Borrowed(original),
                prompt_tokens: original_prompt_tokens,
                maximum_output: original_output,
            });
        }
        Ok(())
    }

    fn output_for(&self, prompt_tokens: NonZeroUsize) -> Option<NonZeroUsize> {
        NonZeroUsize::new(
            self.configured_output
                .get()
                .min(self.context.checked_sub(prompt_tokens.get())?),
        )
    }

    fn reserve_variant(&mut self) -> Result<()> {
        let required = self
            .variants
            .len()
            .checked_add(1)
            .ok_or_else(capacity_error)?;
        if required
            .checked_mul(std::mem::size_of::<ContextVariant<'a>>())
            .is_none_or(|bytes| bytes > self.maximum_inventory_bytes)
        {
            return Err(capacity_error());
        }
        self.variants
            .try_reserve_exact(1)
            .map_err(|_| capacity_error())?;
        if self.inventory_retained_bytes() > self.maximum_inventory_bytes {
            return Err(capacity_error());
        }
        Ok(())
    }
}

fn capacity_error() -> FerrumError {
    FerrumError::resource_exhausted("probe context inventory exceeds shared retained allowance")
}

/// Freeze product-rendered first-suffix and prefill spans before measured work.
/// These variants preserve the original endpoint/policy root identity.
pub(super) fn expand(
    inputs: &mut super::PreparedProbeInputs,
    tokenizer: &dyn Tokenizer,
) -> Result<()> {
    use crate::AutomaticCostProbePromptLimits;
    let release = NonZeroUsize::new(inputs.pair.clean.token_ids.len())
        .ok_or_else(|| FerrumError::invalid_request("empty prepared prefix"))?;
    let maximum_variants = inputs
        .templates
        .len()
        .checked_mul(
            inputs
                .required_geometry
                .sequence_tokens
                .len()
                .checked_add(2)
                .and_then(|n| n.checked_add(inputs.prefill_candidate_chunks.len()))
                .ok_or_else(capacity_error)?,
        )
        .ok_or_else(capacity_error)?;
    let vector_bytes = maximum_variants
        .checked_mul(
            std::mem::size_of::<ContextVariant<'_>>()
                + std::mem::size_of::<AutomaticCostProbeTemplate>()
                + 3 * std::mem::size_of::<usize>(),
        )
        .ok_or_else(capacity_error)?;
    let window_capacity = inputs
        .templates
        .len()
        .checked_mul(inputs.prefill_candidate_chunks.len())
        .ok_or_else(capacity_error)?;
    let window_bytes = window_capacity
        .checked_mul(std::mem::size_of::<super::PrefillCandidateWindow>())
        .ok_or_else(capacity_error)?;
    let retained_roots = inputs
        .templates
        .iter()
        .try_fold(0usize, |n, template| {
            n.checked_add(template.retained_payload_bytes()?)
        })
        .ok_or_else(capacity_error)?;
    let remaining = inputs
        .population
        .maximum_retained_numeric_bytes
        .checked_sub(retained_roots)
        .and_then(|n| n.checked_sub(inputs.required_geometry.retained_payload_bytes()?))
        .and_then(|n| n.checked_sub(vector_bytes))
        .and_then(|n| n.checked_sub(window_bytes))
        .and_then(NonZeroUsize::new)
        .ok_or_else(capacity_error)?;
    let mut budget = AutomaticCostProbePromptBudget::new(AutomaticCostProbePromptLimits {
        maximum_attempts: inputs.settings.maximum_probe_requests,
        maximum_rendered_bytes: remaining,
        maximum_tokenized_tokens: NonZeroUsize::new(
            remaining.get() / std::mem::size_of::<ferrum_types::TokenId>(),
        )
        .ok_or_else(capacity_error)?,
        maximum_retained_bytes: remaining,
    });
    let mut inventory = ContextVariantInventory::new(
        inputs.context,
        inputs.settings.maximum_output_tokens,
        vector_bytes,
    )?;
    for (index, template) in inputs.templates.iter().enumerate() {
        let prompt = NonZeroUsize::new(inputs.prompts[index]).ok_or_else(capacity_error)?;
        let original = inputs.original_template_indices[index];
        let first = prompt
            .get()
            .checked_add(release.get())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(capacity_error)?;
        // Register the original even if its first frontier is not a provider
        // boundary. No new text or renderer work is needed for this point.
        inventory.resolve_first_suffix(
            original,
            template,
            prompt,
            release,
            first,
            tokenizer,
            &mut budget,
        )?;
        // The short standalone source uses one row. A per-row scheduling
        // ceiling does not shrink the independent whole-wave width budget.
        let continuation_chunk = crate::continuous_engine::inner::calibration::geometry_projection::prefill_chunk_for_width(
            inputs.chunk,
            inputs.prefill_row_ceiling,
            1,
        )
        .ok_or_else(capacity_error)?;
        if let ContextVariantResolution::Gap { reason } = inventory.resolve_continuation_prefill(
            original,
            template,
            prompt,
            NonZeroUsize::new(continuation_chunk.get() as usize).ok_or_else(capacity_error)?,
            tokenizer,
            &mut budget,
        )? {
            tracing::warn!(
                original_template_index = original,
                whole_prefill_chunk = inputs.chunk.get(),
                prefill_row_ceiling = inputs.prefill_row_ceiling.map(NonZeroU32::get),
                continuation_row_chunk = continuation_chunk.get(),
                ?reason,
                "Automatic probe product prompt cannot realize short continuation prefill spans"
            );
        }
        for &sequence_tokens in &inputs.required_geometry.sequence_tokens {
            if let ContextVariantResolution::Gap { reason } = inventory.resolve_first_suffix(
                original,
                template,
                prompt,
                release,
                sequence_tokens,
                tokenizer,
                &mut budget,
            )? {
                tracing::warn!(
                    original_template_index = original,
                    sequence_tokens,
                    ?reason,
                    "Automatic probe product prompt cannot realize required first suffix frontier"
                );
            }
        }
    }
    // Keep the original decode/context endpoint product intact. Extra windows
    // only feed explicit prefill-span cases, so widening the finite scheduler
    // chunk menu cannot multiply every decode/width/policy scenario.
    let base_template_count = inventory.variants().len();

    let mut continuation_windows = Vec::new();
    continuation_windows
        .try_reserve_exact(window_capacity)
        .map_err(|_| capacity_error())?;
    if continuation_windows
        .capacity()
        .checked_mul(std::mem::size_of::<super::PrefillCandidateWindow>())
        .is_none_or(|bytes| bytes > window_bytes)
    {
        return Err(capacity_error());
    }
    for (index, template) in inputs.templates.iter().enumerate() {
        let prompt = NonZeroUsize::new(inputs.prompts[index]).ok_or_else(capacity_error)?;
        let original = inputs.original_template_indices[index];
        for &chunk in &inputs.prefill_candidate_chunks {
            match inventory.resolve_continuation_prefill(
                original,
                template,
                prompt,
                NonZeroUsize::new(chunk.get() as usize).ok_or_else(capacity_error)?,
                tokenizer,
                &mut budget,
            )? {
                ContextVariantResolution::Covered { variant_index } => {
                    continuation_windows.push(super::PrefillCandidateWindow {
                        template: variant_index,
                        chunk,
                    });
                }
                ContextVariantResolution::Gap { reason } => {
                    tracing::debug!(
                        original_template_index = original,
                        candidate_chunk = chunk.get(),
                        ?reason,
                        "Automatic declared prefill candidate has no product input window"
                    );
                }
            }
        }
    }
    let mut templates = Vec::with_capacity(inventory.variants().len());
    let mut prompts = Vec::with_capacity(inventory.variants().len());
    let mut outputs = Vec::with_capacity(inventory.variants().len());
    let mut original_indices = Vec::with_capacity(inventory.variants().len());
    for variant in inventory.variants() {
        templates.push(variant.template.as_ref().clone());
        prompts.push(variant.prompt_tokens.get());
        outputs.push(variant.maximum_output);
        original_indices.push(variant.original_template_index);
    }
    tracing::info!(original_templates = inputs.templates.len(), variants = templates.len(),
        prompt_search = ?budget.usage(), "Automatic probe product context variants frozen");
    drop(inventory);
    inputs.base_template_count = base_template_count;
    inputs.continuation_windows = continuation_windows;
    inputs.templates = templates;
    inputs.prompts = prompts;
    inputs.outputs = outputs;
    inputs.original_template_indices = original_indices;
    Ok(())
}

fn covers(
    prompt: NonZeroUsize,
    output: NonZeroUsize,
    release: NonZeroUsize,
    interval: DecodeInterval,
) -> bool {
    release < output
        && prompt
            .get()
            .checked_add(release.get())
            .is_some_and(|start| start <= interval.first_sequence_tokens as usize)
        && prompt
            .get()
            .checked_add(output.get() - 1)
            .is_some_and(|end| end >= interval.last_sequence_tokens as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AutomaticCostProbeOutput, AutomaticCostProbePromptLimits, AutomaticCostProbePromptRenderer,
    };
    use ferrum_types::{InferenceRequest, SpecialTokens, TokenId};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Default)]
    struct Words {
        specials: SpecialTokens,
        stride: usize,
    }
    impl Tokenizer for Words {
        fn encode(&self, text: &str, _: bool) -> Result<Vec<TokenId>> {
            Ok(vec![
                TokenId::new(1);
                text.split_whitespace().count() * self.stride.max(1)
            ])
        }
        fn decode(&self, _: &[TokenId], _: bool) -> Result<String> {
            unreachable!()
        }
        fn decode_incremental(&self, _: &[TokenId], _: TokenId) -> Result<String> {
            unreachable!()
        }
        fn vocab_size(&self) -> usize {
            2
        }
        fn special_tokens(&self) -> &SpecialTokens {
            &self.specials
        }
        fn token_id(&self, _: &str) -> Option<TokenId> {
            None
        }
        fn token_text(&self, _: TokenId) -> Option<&str> {
            None
        }
        fn info(&self) -> ferrum_interfaces::tokenizer::TokenizerInfo {
            unreachable!()
        }
    }
    #[derive(Debug)]
    struct Renderer(Arc<AtomicUsize>);
    impl AutomaticCostProbePromptRenderer for Renderer {
        fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(template(text))
        }
        fn retained_payload_bytes(&self) -> Option<usize> {
            Some(std::mem::size_of::<Self>())
        }
    }
    fn n(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }
    fn template(prompt: &str) -> AutomaticCostProbeTemplate {
        let mut request = InferenceRequest::new(prompt, "fixture");
        request.stream = true;
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap()
    }
    fn budget() -> AutomaticCostProbePromptBudget {
        AutomaticCostProbePromptBudget::new(AutomaticCostProbePromptLimits {
            maximum_attempts: n(64),
            maximum_rendered_bytes: n(65_536),
            maximum_tokenized_tokens: n(16_384),
            maximum_retained_bytes: n(65_536),
        })
    }
    fn interval(first: u32, last: u32) -> DecodeInterval {
        DecodeInterval {
            first_sequence_tokens: first,
            last_sequence_tokens: last,
        }
    }
    fn covered(result: ContextVariantResolution) -> usize {
        let ContextVariantResolution::Covered { variant_index } = result else {
            panic!("covered interval")
        };
        variant_index
    }

    fn continuation<'a>(
        inventory: &mut ContextVariantInventory<'a>,
        original: &'a AutomaticCostProbeTemplate,
        prompt: usize,
        chunk: usize,
        allowance: &mut AutomaticCostProbePromptBudget,
    ) -> Result<ContextVariantResolution> {
        inventory.resolve_continuation_prefill(
            0,
            original,
            n(prompt),
            n(chunk),
            &Words::default(),
            allowance,
        )
    }

    #[test]
    fn continuation_prompt_variant_uses_per_row_ceiling_with_larger_wave_capacity() {
        let original = template("word")
            .with_prompt_renderer(Arc::new(Renderer(Arc::new(AtomicUsize::new(0)))));
        let mut inventory = ContextVariantInventory::new(256, n(32), 16_384).unwrap();
        let mut allowance = budget();
        let chunk = crate::continuous_engine::inner::calibration::geometry_projection::prefill_chunk_for_width(
            NonZeroU32::new(32).unwrap(), NonZeroU32::new(1), 1,
        ).unwrap();
        let index = covered(
            continuation(
                &mut inventory,
                &original,
                1,
                chunk.get() as usize,
                &mut allowance,
            )
            .unwrap(),
        );
        let variant = &inventory.variants()[index];
        assert_eq!(variant.prompt_tokens.get(), 3);
        assert_eq!(
            Words::default()
                .encode(variant.template.prompt(), true)
                .unwrap()
                .len(),
            3
        );
        assert_eq!(variant.original_template_index, 0);
        assert_eq!(variant.maximum_output.get(), 32);
        assert!(allowance.usage().attempts > 0);
    }

    #[test]
    fn continuation_prompt_variant_uses_actual_chunk_and_preserves_original_policy() {
        for chunk in [3, 11, 32] {
            let calls = Arc::new(AtomicUsize::new(0));
            let original = template("word").with_prompt_renderer(Arc::new(Renderer(calls.clone())));
            let mut inventory = ContextVariantInventory::new(256, n(32), 16_384).unwrap();
            let mut allowance = budget();
            let index =
                covered(continuation(&mut inventory, &original, 1, chunk, &mut allowance).unwrap());
            let variant = &inventory.variants()[index];
            let prompt = variant.prompt_tokens.get();
            assert!((2 * chunk + 1..=3 * chunk).contains(&prompt));
            assert_eq!(prompt.div_ceil(chunk), 3);
            assert_eq!(variant.original_template_index, 0);
            assert_eq!(variant.maximum_output.get(), 32.min(256 - prompt));
            assert!(matches!(variant.template, Cow::Owned(_)));
            assert_eq!(
                serde_json::to_value(&variant.template.resolved_request().unwrap().sampling_params)
                    .unwrap(),
                serde_json::to_value(&original.resolved_request().unwrap().sampling_params)
                    .unwrap(),
            );
            assert_eq!(variant.template.output(), original.output());
            assert_eq!(variant.template.response_model(), original.response_model());
            assert_eq!(
                Words::default()
                    .encode(variant.template.prompt(), true)
                    .unwrap()
                    .len(),
                prompt
            );
            assert!(calls.load(Ordering::Relaxed) > 0);
            assert!(allowance.usage().retained_bytes > 0);
        }
    }

    #[test]
    fn continuation_prompt_variant_reuses_frozen_root_and_keeps_long_decode_geometry() {
        let original = template("word")
            .with_prompt_renderer(Arc::new(Renderer(Arc::new(AtomicUsize::new(0)))));
        let mut inventory = ContextVariantInventory::new(256, n(32), 16_384).unwrap();
        let mut allowance = budget();
        let short =
            covered(continuation(&mut inventory, &original, 1, 32, &mut allowance).unwrap());
        let short_prompt = inventory.variants()[short].prompt_tokens;
        let first = u32::try_from(short_prompt.get() + 2).unwrap();
        let prior = allowance.usage();
        assert_eq!(
            covered(
                inventory
                    .resolve_first_suffix(
                        0,
                        &original,
                        n(1),
                        n(2),
                        first,
                        &Words::default(),
                        &mut allowance,
                    )
                    .unwrap()
            ),
            short
        );
        assert_eq!(allowance.usage(), prior);
        let long = covered(
            inventory
                .resolve_first_suffix(
                    0,
                    &original,
                    n(1),
                    n(2),
                    255,
                    &Words::default(),
                    &mut allowance,
                )
                .unwrap(),
        );
        assert_ne!(short, long);
        assert_eq!(inventory.variants()[long].prompt_tokens.get(), 253);
        assert_eq!(inventory.variants()[short].prompt_tokens, short_prompt);
        assert_eq!(inventory.variants().len(), 3);
        assert!(matches!(inventory.variants()[0].template, Cow::Borrowed(_)));
        let prior = allowance.usage();
        assert_eq!(
            covered(continuation(&mut inventory, &original, 1, 32, &mut allowance).unwrap()),
            short
        );
        assert_eq!(allowance.usage(), prior);
    }

    #[test]
    fn continuation_prompt_variant_clips_context_and_reports_unreachable_windows() {
        let calls = Arc::new(AtomicUsize::new(0));
        let original = template("word").with_prompt_renderer(Arc::new(Renderer(calls.clone())));
        let mut allowance = budget();
        let mut inventory = ContextVariantInventory::new(66, n(32), 16_384).unwrap();
        let index =
            covered(continuation(&mut inventory, &original, 1, 32, &mut allowance).unwrap());
        assert_eq!(inventory.variants()[index].prompt_tokens.get(), 65);
        assert_eq!(inventory.variants()[index].maximum_output.get(), 1);
        let mut inventory = ContextVariantInventory::new(65, n(32), 16_384).unwrap();
        let prior = allowance.usage();
        let prior_calls = calls.load(Ordering::Relaxed);
        assert_eq!(
            continuation(&mut inventory, &original, 1, 32, &mut allowance).unwrap(),
            ContextVariantResolution::Gap {
                reason: ContextVariantGap::NoContinuationPrefillWindow
            },
        );
        assert_eq!(allowance.usage(), prior);
        assert_eq!(calls.load(Ordering::Relaxed), prior_calls);
        assert!(continuation(&mut inventory, &original, 1, usize::MAX, &mut allowance).is_err());
        assert_eq!(allowance.usage(), prior);
    }

    #[test]
    fn continuation_prompt_variant_unavailable_renderer_stays_a_typed_gap() {
        for has_renderer in [false, true] {
            let mut original = template("word");
            if has_renderer {
                original = original
                    .with_prompt_renderer(Arc::new(Renderer(Arc::new(AtomicUsize::new(0)))));
            }
            let tokenizer = Words {
                stride: 100,
                ..Default::default()
            };
            let mut inventory = ContextVariantInventory::new(256, n(32), 16_384).unwrap();
            assert_eq!(
                inventory
                    .resolve_continuation_prefill(
                        0,
                        &original,
                        n(100),
                        n(11),
                        &tokenizer,
                        &mut budget(),
                    )
                    .unwrap(),
                ContextVariantResolution::Gap {
                    reason: ContextVariantGap::RendererUnavailableOrUnreachableWindow
                }
            );
            assert_eq!(inventory.variants().len(), 1);
        }
    }

    #[test]
    fn continuation_prompt_variant_bounds_storage_and_binds_original_before_rendering() {
        let calls = Arc::new(AtomicUsize::new(0));
        let original = template("word").with_prompt_renderer(Arc::new(Renderer(calls.clone())));
        let mut inventory =
            ContextVariantInventory::new(256, n(32), std::mem::size_of::<ContextVariant<'_>>())
                .unwrap();
        let mut allowance = budget();
        assert!(continuation(&mut inventory, &original, 1, 32, &mut allowance).is_err());
        assert_eq!(inventory.variants().len(), 1);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(allowance.usage(), Default::default());
        let different = template("two words");
        assert!(continuation(&mut inventory, &different, 2, 32, &mut allowance).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn continuation_prompt_variant_cannot_renew_shared_prompt_allowance() {
        let calls = Arc::new(AtomicUsize::new(0));
        let original = template("word").with_prompt_renderer(Arc::new(Renderer(calls.clone())));
        let mut inventory = ContextVariantInventory::new(256, n(32), 16_384).unwrap();
        let mut allowance = AutomaticCostProbePromptBudget::new(AutomaticCostProbePromptLimits {
            maximum_attempts: n(1),
            maximum_rendered_bytes: n(65_536),
            maximum_tokenized_tokens: n(16_384),
            maximum_retained_bytes: n(65_536),
        });
        assert!(continuation(&mut inventory, &original, 1, 32, &mut allowance).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(allowance.usage().attempts, 2);
        assert_eq!(allowance.usage().retained_bytes, 0);
        assert_eq!(inventory.variants().len(), 1);
    }

    #[test]
    fn context_variants_first_suffix_does_not_count_a_later_reachable_wave() {
        let calls = Arc::new(AtomicUsize::new(0));
        let original =
            template(&"word ".repeat(70)).with_prompt_renderer(Arc::new(Renderer(calls.clone())));
        let mut inventory = ContextVariantInventory::new(96, n(32), 16_384).unwrap();
        let mut budget = budget();
        let index = covered(
            inventory
                .resolve_first_suffix(
                    0,
                    &original,
                    n(70),
                    n(2),
                    95,
                    &Words::default(),
                    &mut budget,
                )
                .unwrap(),
        );
        let variant = &inventory.variants()[index];
        assert_eq!(variant.prompt_tokens.get(), 93);
        assert_eq!(variant.maximum_output.get(), 3);
        assert!(matches!(variant.template, Cow::Owned(_)));
        let usage = budget.usage();
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_eq!(
            covered(
                inventory
                    .resolve_first_suffix(
                        0,
                        &original,
                        n(70),
                        n(2),
                        95,
                        &Words::default(),
                        &mut budget,
                    )
                    .unwrap()
            ),
            index
        );
        assert_eq!(budget.usage(), usage);
    }

    #[test]
    fn context_variants_first_suffix_reports_unreachable_exact_token_frontier() {
        let original = template("word")
            .with_prompt_renderer(Arc::new(Renderer(Arc::new(AtomicUsize::new(0)))));
        let tokenizer = Words {
            stride: 3,
            ..Default::default()
        };
        let mut inventory = ContextVariantInventory::new(96, n(32), 16_384).unwrap();
        assert_eq!(
            inventory
                .resolve_first_suffix(0, &original, n(3), n(2), 94, &tokenizer, &mut budget(),)
                .unwrap(),
            ContextVariantResolution::Gap {
                reason: ContextVariantGap::RendererUnavailableOrUnreachableWindow,
            }
        );
        assert_eq!(inventory.variants().len(), 1);
    }

    #[test]
    fn context_variants_reuse_original_and_recompute_actual_output_room() {
        let original = template(&"word ".repeat(90));
        let mut inventory = ContextVariantInventory::new(96, n(32), 16_384).unwrap();
        let mut budget = budget();
        let index = covered(
            inventory
                .resolve(
                    7,
                    &original,
                    n(90),
                    n(2),
                    interval(94, 95),
                    &Words::default(),
                    &mut budget,
                )
                .unwrap(),
        );
        let again = covered(
            inventory
                .resolve(
                    7,
                    &original,
                    n(90),
                    n(3),
                    interval(94, 95),
                    &Words::default(),
                    &mut budget,
                )
                .unwrap(),
        );
        assert_eq!(index, again);
        assert_eq!(inventory.variants().len(), 1);
        let variant = &inventory.variants()[index];
        assert_eq!(variant.original_template_index, 7);
        assert_eq!(variant.maximum_output.get(), 6);
        assert!(matches!(variant.template, Cow::Borrowed(_)));
        assert_eq!(budget.usage(), Default::default());
        assert!(inventory.inventory_retained_bytes() > 0);
    }

    #[test]
    fn context_variants_reuse_rendered_prompt_across_intervals_and_releases() {
        let calls = Arc::new(AtomicUsize::new(0));
        let original = template("word").with_prompt_renderer(Arc::new(Renderer(calls.clone())));
        let mut inventory = ContextVariantInventory::new(96, n(32), 16_384).unwrap();
        let mut budget = budget();
        let tokenizer = Words {
            stride: 3,
            ..Default::default()
        };
        let index = covered(
            inventory
                .resolve(
                    2,
                    &original,
                    n(3),
                    n(2),
                    interval(94, 95),
                    &tokenizer,
                    &mut budget,
                )
                .unwrap(),
        );
        let usage = budget.usage();
        let rendered_calls = calls.load(Ordering::Relaxed);
        let again = covered(
            inventory
                .resolve(
                    2,
                    &original,
                    n(3),
                    n(4),
                    interval(93, 94),
                    &tokenizer,
                    &mut budget,
                )
                .unwrap(),
        );
        assert_eq!(index, again);
        assert_eq!(budget.usage(), usage);
        assert_eq!(calls.load(Ordering::Relaxed), rendered_calls);
        assert!(rendered_calls > 0);
        let variant = &inventory.variants()[index];
        assert_eq!(variant.original_template_index, 2);
        assert!(matches!(variant.template, Cow::Owned(_)));
        assert_eq!(
            variant.prompt_tokens.get() + variant.maximum_output.get(),
            96
        );
        assert!(variant.maximum_output.get() < 32);
        assert_eq!(inventory.variants().len(), 2);
    }

    #[test]
    fn context_variants_missing_renderer_and_unreachable_context_are_explicit_gaps() {
        let original = template("word");
        let mut inventory = ContextVariantInventory::new(96, n(16), 16_384).unwrap();
        let mut budget = budget();
        for (release, target, reason) in [
            (
                2,
                interval(63, 64),
                ContextVariantGap::RendererUnavailableOrUnreachableWindow,
            ),
            (
                16,
                interval(63, 64),
                ContextVariantGap::NoPostReleaseDecodeWindow,
            ),
            (
                2,
                interval(95, 96),
                ContextVariantGap::IntervalOutsideContext,
            ),
        ] {
            assert_eq!(
                inventory
                    .resolve(
                        0,
                        &original,
                        n(1),
                        n(release),
                        target,
                        &Words::default(),
                        &mut budget
                    )
                    .unwrap(),
                ContextVariantResolution::Gap { reason }
            );
        }
        assert_eq!(budget.usage(), Default::default());
        assert_eq!(inventory.variants().len(), 1);
    }

    #[test]
    fn context_variants_bound_inventory_before_rendering_and_bind_original_root() {
        let calls = Arc::new(AtomicUsize::new(0));
        let original = template("word").with_prompt_renderer(Arc::new(Renderer(calls.clone())));
        let mut inventory =
            ContextVariantInventory::new(96, n(16), std::mem::size_of::<ContextVariant<'_>>())
                .unwrap();
        let mut budget = budget();
        assert!(inventory
            .resolve(
                0,
                &original,
                n(1),
                n(2),
                interval(63, 64),
                &Words::default(),
                &mut budget
            )
            .is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let different = template("two words");
        assert!(inventory
            .resolve(
                0,
                &different,
                n(2),
                n(2),
                interval(4, 5),
                &Words::default(),
                &mut budget
            )
            .is_err());
        assert_eq!(inventory.variants().len(), 1);
    }
}
