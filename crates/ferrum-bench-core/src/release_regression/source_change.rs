//! Small content proofs for production scope. Unrecognized syntax is conservative.
use quote::ToTokens;
use syn::{Attribute, Item, Meta};

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
