//! Small content proofs for production scope. Unrecognized syntax is conservative.
use quote::ToTokens;
use syn::{Attribute, Item, Meta};

mod ready;
pub use ready::run_ready_capability_only;

fn test_configuration(attribute: &Attribute) -> bool {
    let Meta::List(list) = &attribute.meta else {
        return false;
    };
    list.path.is_ident("cfg")
        && syn::parse2::<syn::Ident>(list.tokens.clone()).is_ok_and(|ident| ident == "test")
}

/// Compare Rust ASTs after removing only modules explicitly gated by cfg(test).
/// Nested macro expansion, cfg_attr and complex cfg expressions are not guessed.
/// Comments/formatting disappear through parsing; string literals and production
/// tokens remain in the comparison, including strings resembling test modules.
pub fn rust_validation_only(before: &str, after: &str) -> Result<bool, String> {
    let production = |source: &str| -> Result<String, String> {
        let mut file =
            syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
        file.items.retain(|item| !matches!(item, Item::Mod(module) if module.attrs.iter().any(test_configuration)));
        Ok(file.into_token_stream().to_string())
    };
    Ok(production(before)? == production(after)?)
}

/// Prove that the bench-core crate root only adds its two reviewed release-tool
/// module exports. The caller must restrict this proof to that exact crate root.
/// Existing declarations cannot be removed or altered, and attributes other than
/// documentation are not ignored. All other items retain their complete AST.
pub fn bench_release_exports_only(before: &str, after: &str) -> Result<bool, String> {
    let before = syn::parse_file(before).map_err(|error| format!("Rust scope parse: {error}"))?;
    let mut after = syn::parse_file(after).map_err(|error| format!("Rust scope parse: {error}"))?;
    let mut additions = Vec::new();
    for name in ["release_candidate", "release_regression"] {
        if before
            .items
            .iter()
            .any(|item| matches!(item, Item::Mod(module) if module.ident == name))
        {
            continue;
        }
        let declarations: Vec<_> = after
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                matches!(item, Item::Mod(module) if module.ident == name).then_some(index)
            })
            .collect();
        if declarations.len() > 1 {
            return Ok(false);
        }
        let Some(index) = declarations.first().copied() else {
            continue;
        };
        let Item::Mod(module) = &after.items[index] else {
            unreachable!()
        };
        let documentation_only = module.attrs.iter().all(|attribute| {
            matches!(&attribute.meta, Meta::NameValue(value) if value.path.is_ident("doc")
                && matches!(&value.value, syn::Expr::Lit(literal) if matches!(&literal.lit, syn::Lit::Str(_))))
        });
        if !matches!(module.vis, syn::Visibility::Public(_))
            || module.content.is_some()
            || module.semi.is_none()
            || !documentation_only
        {
            return Ok(false);
        }
        additions.push(index);
    }
    if additions.is_empty() {
        return Ok(false);
    }
    let mut index = 0;
    after.items.retain(|_| {
        let retain = !additions.contains(&index);
        index += 1;
        retain
    });
    Ok(before.into_token_stream().to_string() == after.into_token_stream().to_string())
}

/// Prove a change is confined to the legacy MetalContext submission lifecycle.
/// The caller must restrict this to backend/metal/mod.rs. The independent vNext
/// runtime owns its queues and does not use MetalContext; its own source changes
/// still contribute unrestricted Metal reach. Keep every shared declaration,
/// Backend trait implementation, operator body and buffer representation intact.
/// This is a reach proof, not proof of correct synchronization or performance.
pub fn legacy_metal_submission_only(before: &str, after: &str) -> Result<bool, String> {
    fn function_header(vis: &syn::Visibility, sig: &syn::Signature) -> String {
        let mut sig = sig.clone();
        sig.inputs.pop_punct();
        quote::quote!(#vis #sig).to_string()
    }
    fn reviewed_function(
        attrs: &[Attribute],
        vis: &syn::Visibility,
        sig: &syn::Signature,
        expected: &str,
    ) -> bool {
        let expected: syn::ItemFn = syn::parse_str(expected).expect("fixed scope signature");
        attrs
            .iter()
            .all(|attribute| attribute.path().is_ident("doc") || attribute.path().is_ident("allow"))
            && function_header(vis, sig) == function_header(&expected.vis, &expected.sig)
    }
    fn retained(source: &str) -> Result<Option<String>, String> {
        let mut file =
            syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
        let mut kept = Vec::new();
        let mut flushes = 0;
        let mut seen = std::collections::BTreeSet::new();
        for mut item in file.items {
            if matches!(&item, Item::Mod(module) if module.attrs.iter().any(test_configuration)) {
                continue;
            }
            match &mut item {
                Item::Impl(implementation)
                    if implementation.trait_.is_none()
                        && implementation.generics.params.is_empty()
                        && implementation.generics.where_clause.is_none()
                        && implementation.attrs.is_empty()
                        && implementation.unsafety.is_none()
                        && implementation.defaultness.is_none() =>
                {
                    let owner = implementation.self_ty.to_token_stream().to_string();
                    if owner == "MetalContext" || owner == "MetalBackend" {
                        let mut methods = Vec::new();
                        let mut removed = false;
                        for method in std::mem::take(&mut implementation.items) {
                            let expected = if let syn::ImplItem::Fn(function) = &method {
                                match (owner.as_str(), function.sig.ident.to_string().as_str()) {
                                    ("MetalContext", "flush") => Some("pub(crate) fn flush(&mut self) {}"),
                                    ("MetalContext", "submit_and_wait") => Some("fn submit_and_wait(&mut self) -> Option<&'static metal::CommandBufferRef> {}"),
                                    ("MetalContext", "flush_checked") => Some("fn flush_checked(&mut self) -> Result<()> {}"),
                                    ("MetalBackend", "sync_checked") => Some("pub fn sync_checked(ctx: &mut MetalContext) -> Result<()> {}"),
                                    _ => None,
                                }
                            } else {
                                None
                            };
                            if let Some(expected) = expected {
                                let syn::ImplItem::Fn(function) = &method else {
                                    unreachable!()
                                };
                                if !reviewed_function(
                                    &function.attrs,
                                    &function.vis,
                                    &function.sig,
                                    expected,
                                ) || !seen.insert(format!("{owner}::{}", function.sig.ident))
                                {
                                    return Ok(None);
                                }
                                if owner == "MetalContext" && function.sig.ident == "flush" {
                                    flushes += 1;
                                }
                                removed = true;
                            } else {
                                methods.push(method);
                            }
                        }
                        implementation.items = methods;
                        // Only the reviewed checked-sync API may introduce an
                        // otherwise empty inherent Backend implementation.
                        if owner == "MetalBackend" && removed && implementation.items.is_empty() {
                            continue;
                        }
                    }
                }
                Item::Fn(function) => {
                    let expected = match function.sig.ident.to_string().as_str() {
                        "command_buffer_error" => Some("fn command_buffer_error(cmd: &metal::CommandBufferRef) -> (Option<i64>, Option<String>) {}"),
                        "validate_command_buffer_completion" => Some("fn validate_command_buffer_completion(status: metal::MTLCommandBufferStatus, code: Option<i64>, detail: Option<&str>) -> Result<()> {}"),
                        _ => None,
                    };
                    if let Some(expected) = expected {
                        if !reviewed_function(
                            &function.attrs,
                            &function.vis,
                            &function.sig,
                            expected,
                        ) || !seen.insert(function.sig.ident.to_string())
                        {
                            return Ok(None);
                        }
                        continue;
                    }
                }
                _ => {}
            }
            kept.push(item);
        }
        if flushes != 1 {
            return Ok(None);
        }
        file.items = kept;
        Ok(Some(file.into_token_stream().to_string()))
    }
    Ok(
        matches!((retained(before)?, retained(after)?), (Some(before), Some(after)) if before == after),
    )
}

/// The repository's explicit Homebrew blocks are the first shell block in
/// Quick Start and Installation. Everything outside these blocks and their
/// installation introduction must remain unchanged, including all model commands,
/// features, performance promises and the tarball/Cargo installation instructions.
/// This classification requires installation checks; it does not approve prose.
pub fn homebrew_documentation_only(before: &str, after: &str) -> Result<bool, String> {
    fn without_homebrew(text: &str) -> Result<Vec<String>, String> {
        let lines: Vec<_> = text.lines().collect();
        let mut regions = Vec::new();
        for headings in [
            ["## Quick Start", "## 快速开始"],
            ["## Installation", "## 安装"],
        ] {
            let starts: Vec<_> = lines
                .iter()
                .enumerate()
                .filter_map(|(index, line)| headings.contains(line).then_some(index))
                .collect();
            if starts.len() != 1 {
                return Err(
                    "README must have one explicit quick-start and installation heading".into(),
                );
            }
            let start = starts[0] + 1;
            let end = (start..lines.len())
                .find(|index| lines[*index].starts_with("## "))
                .unwrap_or(lines.len());
            let fence = (start..end)
                .find(|index| lines[*index] == "```bash")
                .ok_or("installation section is missing a bash block")?;
            let intro = lines[start..fence].join("\n");
            let intro = intro.trim();
            let quick_start = headings[0] == "## Quick Start";
            let recognized_intro = if quick_start {
                intro.starts_with("Install Ferrum:") || intro.starts_with("安装 Ferrum：")
            } else {
                intro.starts_with("Homebrew")
            };
            if !recognized_intro
                || intro.contains("```")
                || intro.lines().any(|line| line.starts_with('#'))
            {
                return Err("unrecognized Homebrew introduction".into());
            }
            let close = (fence + 1..end)
                .find(|index| lines[*index] == "```")
                .ok_or("unterminated Homebrew shell block")?;
            let mut commands = false;
            for line in &lines[fence + 1..close] {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if !line.starts_with("brew ")
                    || line.contains([';', '&', '|', '$', '`', '<', '>', '\\'])
                {
                    return Err(
                        "installation block contains a non-Homebrew or compound command".into(),
                    );
                }
                commands = true;
            }
            if !commands {
                return Err("empty Homebrew command block".into());
            }
            regions.push((start, close + 1));
        }
        let mut retained = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            if !regions
                .iter()
                .any(|(start, end)| (*start..*end).contains(&index))
            {
                retained.push((*line).to_owned());
            }
        }
        Ok(retained)
    }
    Ok(without_homebrew(before)? == without_homebrew(after)?)
}

#[cfg(test)]
#[path = "source_change_tests.rs"]
mod tests;
