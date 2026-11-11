//! Comfy PH3ImageConditioning's edit instruction presentation. This does not
//! perform prompt expansion; it preserves an already expanded edit brief.
use super::*;
use std::{borrow::Cow, sync::OnceLock};
use tokenizers::utils::SysRegex;

impl PH3ImageTextTokenizer {
    /// Apply the model node's edit instruction before image/token geometry is
    /// prepared. The lower-level `prepare_single_image` remains raw text.
    pub fn prepare_single_image_instruction(
        &self,
        prompt: &str,
        grid: &Qwen3VLVisionGrid,
        merge_size: u64,
    ) -> Result<PH3ImageEditPrompt> {
        self.prepare_images_instruction(prompt, std::slice::from_ref(grid), merge_size)
    }

    pub fn prepare_images_instruction(
        &self,
        prompt: &str,
        grids: &[Qwen3VLVisionGrid],
        merge_size: u64,
    ) -> Result<PH3ImageEditPrompt> {
        let text = instruction(prompt, self.limits.max_prompt_bytes)?;
        self.prepare_images(&text, grids, merge_size)
    }
}

fn instruction(prompt: &str, maximum_bytes: usize) -> Result<Cow<'_, str>> {
    ensure!(
        prompt.len() <= maximum_bytes,
        "edit instruction exceeds prompt capacity"
    );
    static VERBATIM: OnceLock<SysRegex> = OnceLock::new();
    let verbatim = VERBATIM.get_or_init(|| {
        SysRegex::new(r"^(Edit <Picture \d+>|subject_definitions:)")
            .expect("fixed Comfy edit instruction expression is valid")
    });
    // Python's regex whitespace and str.strip also include the four C0
    // information separators, beyond Rust's Unicode White_Space predicate.
    let space = |c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}');
    if verbatim
        .find_iter(prompt.trim_start_matches(space))
        .next()
        .is_some()
    {
        return Ok(Cow::Borrowed(prompt.trim_matches(space)));
    }
    const PREFIX: &str = "Edit <Picture 1>. ";
    let size = PREFIX
        .len()
        .checked_add(prompt.len())
        .context("edit instruction extent overflows")?;
    ensure!(
        size <= maximum_bytes,
        "presented edit instruction exceeds prompt capacity"
    );
    Ok(Cow::Owned(format!("{PREFIX}{prompt}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_instruction_preserves_expanded_briefs_and_explicit_picture_targets() {
        for (input, expected) in [
            ("make the sky blue", "Edit <Picture 1>. make the sky blue"),
            (
                "\u{1e}Edit <Picture 1>. Keep the subject.\u{1f}",
                "Edit <Picture 1>. Keep the subject.",
            ),
            (
                "  make the sky blue  ",
                "Edit <Picture 1>.   make the sky blue  ",
            ),
            (
                "\n Edit <Picture 1>. Keep the subject unchanged. \n",
                "Edit <Picture 1>. Keep the subject unchanged.",
            ),
            (
                "Edit <Picture 12>. Keep the other subjects.",
                "Edit <Picture 12>. Keep the other subjects.",
            ),
            (
                "\nsubject_definitions:\n  <Subject 1>: a cat\n",
                "subject_definitions:\n  <Subject 1>: a cat",
            ),
            (
                "Edit <Picture x>. blue",
                "Edit <Picture 1>. Edit <Picture x>. blue",
            ),
            (
                "subject_definitions without colon",
                "Edit <Picture 1>. subject_definitions without colon",
            ),
        ] {
            assert_eq!(instruction(input, 256).unwrap(), expected);
        }
    }

    #[test]
    fn complete_instruction_is_bounded_before_tokenization_and_raw_method_stays_raw() {
        let t = super::super::tests::tokenizer(128, 128);
        let grid = Qwen3VLVisionGrid::new(1, 4, 6).unwrap();
        let presented = t
            .prepare_single_image_instruction("cat after", &grid, 2)
            .unwrap();
        let expected = t
            .prepare_single_image("Edit <Picture 1>. cat after", &grid, 2)
            .unwrap();
        let raw = t.prepare_single_image("cat after", &grid, 2).unwrap();
        assert_eq!(presented.text().token_ids(), expected.text().token_ids());
        assert_ne!(presented.text().token_ids(), raw.text().token_ids());
        let limit = "Edit <Picture 1>. cat after".len();
        assert!(instruction("cat after", limit).is_ok());
        assert!(instruction("cat after", limit - 1).is_err());
        assert!(t
            .prepare_single_image_instruction("<|vision_start|>", &grid, 2)
            .is_err());
        let bounded = super::super::tests::tokenizer(128, raw.text().valid_lengths()[0]);
        assert!(bounded
            .prepare_single_image_instruction("cat after", &grid, 2)
            .is_err());
    }
}
