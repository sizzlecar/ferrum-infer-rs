use super::*;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const MAX_SOURCE_BYTES: usize = 32 * 1024 * 1024;
const MAX_TOTAL_PROMPT_BYTES: usize = 1024 * 1024;

pub struct QwenImage21T2iTokenizer {
    engine: Tokenizer,
    pad: u32,
    forbidden_vision: Vec<u32>,
    prefix: Vec<u32>,
    baseline: usize,
    source_sha256: [String; 3],
}
impl QwenImage21T2iTokenizer {
    /// Accept immutable source bytes from one resolved processor. Actual
    /// system-only template rendering has no generation
    /// prompt and is tokenized without added special tokens, as HF does.
    pub fn from_source_bytes(
        tokenizer: &[u8],
        config: &[u8],
        chat_template: &[u8],
    ) -> Result<Self> {
        ensure!(
            !tokenizer.is_empty() && tokenizer.len() <= MAX_SOURCE_BYTES,
            "tokenizer source must be nonempty and <=32 MiB"
        );
        ensure!(
            !config.is_empty() && config.len() <= 1024 * 1024,
            "tokenizer config must be nonempty and <=1 MiB"
        );
        ensure!(
            chat_template.len() <= 64 * 1024,
            "chat template exceeds 64 KiB"
        );
        let rendered_system = render_system_template(std::str::from_utf8(chat_template)?)?;
        ensure!(
            rendered_system == system_message(),
            "processor system-message semantics differ from the T2I raw template"
        );
        let c: serde_json::Value = serde_json::from_slice(config)?;
        ensure!(
            c["tokenizer_class"] == "Qwen2Tokenizer"
                && c["processor_class"] == "Qwen3VLProcessor"
                && c["bos_token"].is_null()
                && c["pad_token"] == "<|endoftext|>"
                && c["eos_token"] == "<|im_end|>"
                && c["add_prefix_space"] == false
                && c["split_special_tokens"] == false,
            "unsupported processor tokenizer semantics"
        );
        let mut engine = Tokenizer::from_bytes(tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer source: {e}"))?;
        // Processor requests padding explicitly. Do it only once, after
        // untruncated per-sample encoding, preserving valid literal PAD IDs.
        engine.with_padding(None);
        engine
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        let pp = engine
            .get_post_processor()
            .ok_or_else(|| anyhow::anyhow!("missing ByteLevel postprocessor"))?;
        ensure!(
            serde_json::to_value(pp)?["type"] == "ByteLevel",
            "only the shipped non-inserting ByteLevel postprocessor is supported"
        );
        let pad = engine
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("missing PAD token"))?;
        for marker in ["<|im_start|>", "<|im_end|>"] {
            let id = engine
                .token_to_id(marker)
                .ok_or_else(|| anyhow::anyhow!("missing role marker {marker}"))?;
            ensure!(
                encode(&engine, marker, true)? == [id],
                "role marker must encode to exactly its special token"
            );
        }
        let prefix = encode(&engine, &rendered_system, false)?;
        let baseline = encode(&engine, &render(""), true)?.len();
        ensure!(
            !prefix.is_empty() && prefix.len() < baseline,
            "invalid system/template token lengths"
        );
        let forbidden_vision = [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        .into_iter()
        .filter_map(|s| engine.token_to_id(s))
        .collect();
        let hash = |b: &[u8]| format!("{:x}", Sha256::digest(b));
        Ok(Self {
            engine,
            pad,
            forbidden_vision,
            prefix,
            baseline,
            source_sha256: [hash(tokenizer), hash(config), hash(chat_template)],
        })
    }
    /// SHA-256 of tokenizer, tokenizer config and processor template, in order.
    pub fn source_sha256(&self) -> &[String; 3] {
        &self.source_sha256
    }
    pub fn drop_prefix_tokens(&self) -> usize {
        self.prefix.len()
    }
    /// Mirrors Omni's user-token budget: max(full_tokens - empty_template, 0).
    /// Empty strings become one space. No truncation occurs. Raw prompt bytes
    /// across the batch are independently bounded to 1 MiB for tokenizer work.
    pub fn prepare(
        &self,
        prompts: &[&str],
        maximum_user_tokens: usize,
    ) -> Result<QwenImage21T2iPromptBatch> {
        ensure!(
            !prompts.is_empty() && prompts.len() <= 16,
            "T2I requires 1..=16 prompts"
        );
        ensure!(
            (1..=8192).contains(&maximum_user_tokens),
            "user-token budget must be 1..=8192"
        );
        let bytes = prompts
            .iter()
            .try_fold(0usize, |n, p| n.checked_add(p.len()))
            .ok_or_else(|| anyhow::anyhow!("prompt byte count overflows"))?;
        ensure!(
            bytes <= MAX_TOTAL_PROMPT_BYTES,
            "batch raw prompts exceed 1 MiB"
        );
        let samples = prompts
            .iter()
            .map(|prompt| {
                let ids = encode(
                    &self.engine,
                    &render(if prompt.is_empty() { " " } else { prompt }),
                    true,
                )?;
                ensure!(
                    ids.len().saturating_sub(self.baseline) <= maximum_user_tokens,
                    "prompt exceeds user-token budget; truncation is disabled"
                );
                ensure!(
                    ids.starts_with(&self.prefix),
                    "full prompt tokenization changes the system prefix"
                );
                ensure!(
                    !ids.iter().any(|id| self.forbidden_vision.contains(id)),
                    "pure T2I does not accept visual placeholder tokens"
                );
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;
        QwenImage21T2iPromptBatch::from_encoded(samples, self.prefix.len(), self.pad)
    }
}
fn encode(engine: &Tokenizer, text: &str, special: bool) -> Result<Vec<u32>> {
    let e = engine
        .encode(text, special)
        .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
    ensure!(
        e.get_attention_mask().iter().all(|&x| x == 1),
        "unexpected internal tokenizer padding"
    );
    Ok(e.get_ids().to_vec())
}
fn system_message() -> String {
    format!("<|im_start|>system\n{T2I_SYSTEM_PROMPT}<|im_end|>\n")
}
fn render(prompt: &str) -> String {
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_message()
    )
}

fn render_system_template(source: &str) -> Result<String> {
    let mut environment = minijinja::Environment::new();
    environment.set_fuel(Some(100_000));
    environment.set_recursion_limit(64);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    environment.add_template("processor", source)?;
    let mut output = LimitedOutput(Vec::new());
    environment.get_template("processor")?.render_captured_to(minijinja::context! {
        messages => serde_json::json!([{"role":"system","content":[{"type":"text","text":T2I_SYSTEM_PROMPT}]}]),
        tools => Vec::<serde_json::Value>::new(), add_generation_prompt => false,
        add_vision_id => false, bos_token => "", eos_token => "<|im_end|>",
    }, &mut output)?;
    Ok(String::from_utf8(output.0)?)
}
struct LimitedOutput(Vec<u8>);
impl std::io::Write for LimitedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > (64 * 1024usize).saturating_sub(self.0.len()) {
            return Err(std::io::Error::other(
                "rendered system template exceeds 64 KiB",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
