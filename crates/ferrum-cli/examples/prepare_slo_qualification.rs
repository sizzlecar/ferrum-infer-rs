//! Offline synthetic qualification input preparation. No model execution or network.
//!
//! Usage: cargo run -p ferrum-cli --example prepare_slo_qualification -- RECIPE.json NEW_DIR
#[path = "prepare_slo_qualification/generation.rs"]
mod generation;
#[path = "prepare_slo_qualification/io.rs"]
mod io;
#[path = "prepare_slo_qualification/recipe.rs"]
mod recipe;

use anyhow::{bail, Context, Result};
use ferrum_server::chat_template::ModelChatTemplate;
use ferrum_tokenizer::HuggingFaceTokenizer;
#[cfg(test)]
use generation::{bounded_search, nonce};
use generation::{prepare_stage, GenerationBudget};
use io::{publish_new_directory, read_bounded, write_json_bounded};
use recipe::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn token_string(value: &Value, key: &str) -> Option<String> {
    value[key]
        .as_str()
        .or_else(|| value[key]["content"].as_str())
        .map(str::to_owned)
}

fn selected_template(
    recipe: &Recipe,
    tokenizer_config: &[u8],
    jinja: Option<&[u8]>,
) -> Result<(ModelChatTemplate, Vec<u8>)> {
    let config: Value = serde_json::from_slice(tokenizer_config)?;
    let (text, source) = match recipe.template_source {
        TemplateSource::TokenizerConfig => (
            config["chat_template"]
                .as_str()
                .context("tokenizer_config requires an explicitly selected string chat_template")?,
            recipe.model_snapshot.join("tokenizer_config.json"),
        ),
        TemplateSource::ChatTemplateJinja => (
            std::str::from_utf8(jinja.context("explicit chat_template.jinja is missing")?)?,
            recipe.model_snapshot.join("chat_template.jinja"),
        ),
    };
    if text.trim().is_empty() {
        bail!("explicitly selected chat template must not be empty");
    }
    // Product templates otherwise see the current clock. A future frozen-clock
    // recipe can make that input explicit; this preparer never guesses a date.
    if text.contains("strftime_now") {
        bail!("time-dependent template requires a separately frozen clock; unsupported by this recipe");
    }
    let mut template = ModelChatTemplate::new(text, source.display().to_string());
    if matches!(recipe.template_source, TemplateSource::TokenizerConfig) {
        template.bos_token = token_string(&config, "bos_token");
        template.eos_token = token_string(&config, "eos_token");
    }
    template.set_output_protocol(recipe.output_protocol);
    Ok((template, text.as_bytes().to_vec()))
}

async fn prepare(recipe_path: &Path, destination: &Path) -> Result<()> {
    let recipe_bytes = read_bounded(recipe_path, MAX_RECIPE_BYTES)?;
    let recipe: Recipe = serde_json::from_slice(&recipe_bytes)?;
    recipe.validate()?;
    let recipe_hash = hash(&recipe_bytes);
    let snapshot = &recipe.model_snapshot;
    let tokenizer_bytes = read_bounded(
        &snapshot.join("tokenizer.json"),
        recipe.limits.max_source_bytes,
    )?;
    let tokenizer_config = read_bounded(
        &snapshot.join("tokenizer_config.json"),
        recipe.limits.max_source_bytes,
    )?;
    let model_config = read_bounded(
        &snapshot.join("config.json"),
        recipe.limits.max_source_bytes,
    )?;
    let generation_config = match read_bounded(
        &snapshot.join("generation_config.json"),
        recipe.limits.max_source_bytes,
    ) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let jinja = if matches!(recipe.template_source, TemplateSource::ChatTemplateJinja) {
        Some(read_bounded(
            &snapshot.join("chat_template.jinja"),
            recipe.limits.max_source_bytes,
        )?)
    } else {
        None
    };
    let (template, template_bytes) =
        selected_template(&recipe, &tokenizer_config, jinja.as_deref())?;
    let tokenizer = HuggingFaceTokenizer::from_source_bytes(
        &tokenizer_bytes,
        Some(&tokenizer_config),
        generation_config.as_deref(),
    )
    .await?;

    // A sibling staging directory makes the final no-replace rename one atomic
    // publication on supported OSes. A target created during preparation wins.
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if destination.file_name().is_none() {
        bail!("output must name a new directory");
    }
    let staging = tempfile::Builder::new()
        .prefix(".slo-qualification-")
        .tempdir_in(parent)?;
    let mut remaining_output = recipe.limits.max_output_bytes;
    let mut budget = GenerationBudget::new(recipe.limits.max_candidate_bytes);
    let mut retained_payload_bytes = 0usize;
    let mut previous_tokens = Vec::new();
    let mut stage_bindings = Vec::with_capacity(recipe.stages.len());
    for stage in &recipe.stages {
        let prepared = prepare_stage(
            &recipe,
            stage,
            &tokenizer,
            &template,
            &recipe_hash,
            &mut budget,
            &mut retained_payload_bytes,
            &mut previous_tokens,
        )?;
        let client_name = format!("{}.client.json", stage.name);
        let audit_name = format!("{}.prompt-audit.json", stage.name);
        let client_hash = write_json_bounded(
            &staging.path().join(&client_name),
            &prepared.client,
            &mut remaining_output,
        )?;
        let audit_hash = write_json_bounded(
            &staging.path().join(&audit_name),
            &prepared.audit,
            &mut remaining_output,
        )?;
        stage_bindings.push(json!({"name":stage.name,"client_file":client_name,"client_sha256":client_hash,
            "audit_file":audit_name,"audit_sha256":audit_hash,"requests":prepared.client.requests.len()}));
    }
    let recipe_copy_hash = write_json_bounded(
        &staging.path().join("recipe.json"),
        &recipe,
        &mut remaining_output,
    )?;
    let receipt = json!({
        "schema_version":1,"kind":"synthetic_public_geometry_qualification",
        "recipe_input_sha256":recipe_hash,"normalized_recipe_sha256":recipe_copy_hash,
        "model_snapshot":snapshot,"template_source":recipe.template_source,
        "model_config_sha256":hash(&model_config),"tokenizer_sha256":hash(&tokenizer_bytes),
        "tokenizer_config_sha256":hash(&tokenizer_config),
        "generation_config_sha256":generation_config.as_deref().map(hash),
        "selected_template_sha256":hash(&template_bytes),"output_protocol":recipe.output_protocol,
        "encode_add_special_tokens":true,"stages":stage_bindings,
        "evaluations":budget.evaluations,"candidate_bytes_charged":budget.bytes_charged,
        "qualification_claim":"exact prepared input geometry only; actual execution widths, prefix hits, completion, model support and SLOs are unproven",
        "configuration_scope":"context, block size, output protocol and template source are explicit recipe declarations; source hashes do not verify a running server descriptor",
        "resource_scope":"finite request/search/file/output bounds; the template renderer and tokenizer internals have no hard RSS or time sandbox",
        "prefix_scope":"potential shared rendered token prefixes in declared stage/order; no cache-hit or lifetime claim"
    });
    write_json_bounded(
        &staging.path().join("receipt.json"),
        &receipt,
        &mut remaining_output,
    )?;
    publish_new_directory(staging.path(), destination)?;
    // TempDir now refers to the old staging path; dropping it never removes the
    // newly published destination. File flush/sync is distinct from power-loss
    // durability of the parent-directory rename.
    drop(staging);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).take(3).collect();
    if args.len() != 2 {
        bail!("usage: prepare_slo_qualification RECIPE.json NEW_OUTPUT_DIRECTORY");
    }
    prepare(Path::new(&args[0]), Path::new(&args[1])).await
}

#[cfg(test)]
#[path = "prepare_slo_qualification/independent_tests.rs"]
mod independent_tests;

#[cfg(test)]
#[path = "prepare_slo_qualification/search_boundary_tests.rs"]
mod search_boundary_tests;
