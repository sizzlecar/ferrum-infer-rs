//! Preserve the behavior affected by reviewed product dependency changes.
//! Normalize only declared feature/edge transitions, then reuse the complete
//! snapshot verifier for all other manifest and registry inputs.
use super::*;
use crate::release_regression::ChangeArea;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductDependencyRefinement {
    pub paths: BTreeMap<String, Vec<ChangeArea>>,
    pub validation: DependencyRefinement,
}

fn table_at<'a>(document: &'a DocumentMut, keys: &[&str]) -> Option<&'a dyn TableLike> {
    let mut item = document.as_item();
    for key in keys {
        item = item.get(key)?;
    }
    item.as_table_like()
}

fn item_at_mut<'a>(document: &'a mut DocumentMut, keys: &[&str]) -> Option<&'a mut Item> {
    let mut item = document.as_item_mut();
    for key in keys {
        item = item.get_mut(key)?;
    }
    Some(item)
}

fn features(item: &Item) -> Option<BTreeSet<&str>> {
    item.get("features")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect()
}

fn normalize_feature_addition(
    prior: &DocumentMut,
    next: &mut DocumentMut,
    dependency: &str,
    added: &str,
) -> bool {
    let keys = ["workspace", "dependencies", dependency];
    let Some(before) = table_at(prior, &keys[..2]).and_then(|t| t.get(dependency)) else {
        return false;
    };
    let Some(after) = item_at_mut(next, &keys) else {
        return false;
    };
    let (Some(old), Some(new)) = (features(before), features(after)) else {
        return false;
    };
    if old.contains(added) || new != old.union(&BTreeSet::from([added])).copied().collect() {
        return false;
    }
    let mut normalized = after.clone();
    normalized["features"] = before["features"].clone();
    if semantic(&normalized) != semantic(before) {
        return false;
    }
    *after = before.clone();
    true
}

fn normalize_tokenizer_training(prior: &DocumentMut, next: &mut DocumentMut) -> bool {
    let keys = ["workspace", "dependencies", "tokenizers"];
    let Some(before) = table_at(prior, &keys[..2]).and_then(|t| t.get("tokenizers")) else {
        return false;
    };
    let Some(after) = item_at_mut(next, &keys) else {
        return false;
    };
    // tokenizers 0.21 defaults enable progressbar, onig and esaxx_fast. Keep
    // inference tokenization features while disabling the C++ trainer build.
    if before.get("version").and_then(Item::as_str) != Some("0.21")
        || before.get("default-features").is_some()
        || features(before) != Some(BTreeSet::from(["onig"]))
        || after.get("default-features").and_then(Item::as_bool) != Some(false)
        || features(after) != Some(BTreeSet::from(["onig", "progressbar"]))
    {
        return false;
    }
    let mut normalized = after.clone();
    normalized
        .as_table_like_mut()
        .unwrap()
        .remove("default-features");
    normalized["features"] = before["features"].clone();
    if semantic(&normalized) != semantic(before) {
        return false;
    }
    *after = before.clone();
    true
}

fn remove_empty_tables(document: &mut DocumentMut, keys: &[&str]) {
    for depth in (1..=keys.len()).rev() {
        let empty = table_at(document, &keys[..depth]).is_some_and(|table| table.is_empty());
        if empty {
            if depth == 1 {
                document.remove(keys[0]);
            } else if let Some(parent) = item_at_mut(document, &keys[..depth - 1]) {
                parent.as_table_like_mut().unwrap().remove(keys[depth - 1]);
            }
        }
    }
}

fn remove_added_dependency(
    prior: &DocumentMut,
    next: &mut DocumentMut,
    table: &[&str],
    dependency: &str,
    reviewed: impl FnOnce(&Item) -> bool,
) -> bool {
    if table_at(prior, table).is_some_and(|t| t.get(dependency).is_some()) {
        return false;
    }
    let Some(value) = table_at(next, table).and_then(|t| t.get(dependency)) else {
        return false;
    };
    if !reviewed(value) {
        return false;
    }
    item_at_mut(next, table)
        .unwrap()
        .as_table_like_mut()
        .unwrap()
        .remove(dependency);
    remove_empty_tables(next, table);
    true
}

fn process_api_dependency(item: &Item) -> bool {
    let Some(table) = item.as_table_like() else {
        return false;
    };
    let Some(selected) = features(item) else {
        return false;
    };
    let allowed = BTreeSet::from([
        "Win32_Foundation",
        "Win32_Security",
        "Win32_Storage_FileSystem",
        "Win32_System_Console",
        "Win32_System_JobObjects",
        "Win32_System_Threading",
        "Win32_System_ProcessStatus",
    ]);
    table
        .iter()
        .all(|(key, _)| matches!(key, "version" | "features"))
        && item.get("version").and_then(Item::as_str) == Some("0.61.2")
        && !selected.is_empty()
        && selected.is_subset(&allowed)
}

fn object_reader_dependency(item: &Item) -> bool {
    let Some(table) = item.as_table_like() else {
        return false;
    };
    table
        .iter()
        .all(|(key, _)| matches!(key, "version" | "features" | "default-features"))
        && item.get("version").and_then(Item::as_str) == Some("=0.37.3")
        && item.get("default-features").and_then(Item::as_bool) == Some(false)
        && features(item)
            == Some(BTreeSet::from([
                "read_core",
                "archive",
                "coff",
                "unaligned",
                "std",
            ]))
}

fn normalize_launcher_target(prior: &DocumentMut, next: &mut DocumentMut) -> bool {
    let existing = |document: &DocumentMut| {
        document
            .get("bin")
            .and_then(Item::as_array_of_tables)
            .is_some_and(|bins| {
                bins.iter()
                    .any(|bin| bin.get("name").and_then(Item::as_str) == Some("ferrum-launcher"))
            })
    };
    if existing(prior)
        || prior
            .get("package")
            .and_then(|p| p.get("default-run"))
            .is_some()
        || next
            .get("package")
            .and_then(|p| p.get("default-run"))
            .and_then(Item::as_str)
            != Some("ferrum")
    {
        return false;
    }
    let Some(bins) = next.get_mut("bin").and_then(Item::as_array_of_tables_mut) else {
        return false;
    };
    let indices: Vec<_> = bins
        .iter()
        .enumerate()
        .filter_map(|(i, bin)| {
            (bin.get("name").and_then(Item::as_str) == Some("ferrum-launcher")
                && bin.get("path").and_then(Item::as_str) == Some("src/bin/ferrum-launcher.rs")
                && bin.iter().all(|(key, _)| matches!(key, "name" | "path")))
            .then_some(i)
        })
        .collect();
    if indices.len() != 1 {
        return false;
    }
    bins.remove(indices[0]);
    next["package"]
        .as_table_like_mut()
        .unwrap()
        .remove("default-run");
    true
}

fn normalize_nccl_target(prior: &DocumentMut, next: &mut DocumentMut) -> bool {
    let table = [
        "target",
        "cfg(not(target_os = \"windows\"))",
        "dependencies",
    ];
    let Some(before_cuda) = prior.get("features").and_then(|f| f.get("cuda")) else {
        return false;
    };
    let Some(after_cuda) = next.get("features").and_then(|f| f.get("cuda")) else {
        return false;
    };
    let values = |item: &Item| {
        item.as_array().and_then(|a| {
            a.iter()
                .map(Value::as_str)
                .map(|x| x.map(str::to_owned))
                .collect::<Option<BTreeSet<_>>>()
        })
    };
    let (Some(mut old), Some(new)) = (values(before_cuda), values(after_cuda)) else {
        return false;
    };
    if !old.remove("cudarc?/nccl") || old != new {
        return false;
    }
    let accepted = remove_added_dependency(prior, next, &table, "cudarc", |spec| {
        let Some(global) = prior.get("dependencies").and_then(|d| d.get("cudarc")) else {
            return false;
        };
        spec.as_table_like().is_some_and(|t| {
            t.iter()
                .all(|(k, _)| matches!(k, "version" | "default-features" | "features" | "optional"))
        }) && spec.get("version").map(semantic) == global.get("version").map(semantic)
            && spec.get("default-features").and_then(Item::as_bool) == Some(false)
            && spec.get("optional").and_then(Item::as_bool) == Some(true)
            && features(spec) == Some(BTreeSet::from(["nccl"]))
    });
    if accepted {
        next["features"]["cuda"] = before_cuda.clone();
    }
    accepted
}

/// Dependency bodies and registry resolution stay under the existing verifier.
/// An unrecognized product delta rejects the refinement as a whole.
pub fn product_dependency_paths(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> Result<ProductDependencyRefinement, String> {
    let mut normalized = after.clone();
    let mut paths = BTreeMap::new();
    let root = parse(before, "Cargo.toml")?;
    let mut next_root = parse(after, "Cargo.toml")?;
    let grammar = normalize_feature_addition(&root, &mut next_root, "llguidance", "lark");
    let tokenizer = normalize_tokenizer_training(&root, &mut next_root);
    if grammar || tokenizer {
        paths.insert(
            "Cargo.toml".into(),
            vec![
                ChangeArea::Build,
                ChangeArea::Template,
                ChangeArea::Termination,
                ChangeArea::Structured,
                ChangeArea::Tools,
            ],
        );
        normalized.insert("Cargo.toml".into(), next_root.to_string());
    }
    if tokenizer {
        normalize_training_lock(before, &mut normalized)?;
    }
    for (path, package) in [
        ("crates/ferrum-cli/Cargo.toml", "ferrum-cli"),
        ("crates/ferrum-types/Cargo.toml", "ferrum-types"),
        ("crates/ferrum-native-ops/Cargo.toml", "ferrum-native-ops"),
        ("crates/ferrum-sampler/Cargo.toml", "ferrum-sampler"),
        ("crates/ferrum-kernels/Cargo.toml", "ferrum-kernels"),
    ] {
        if !before.contains_key(path) && !after.contains_key(path) {
            continue;
        }
        let prior = parse(before, path)?;
        let mut next = parse(after, path)?;
        let mut areas = BTreeSet::new();
        if matches!(package, "ferrum-cli" | "ferrum-types")
            && remove_added_dependency(
                &prior,
                &mut next,
                &["target", "cfg(windows)", "dependencies"],
                "windows-sys",
                process_api_dependency,
            )
        {
            normalize_lock_edge(before, &mut normalized, package, "windows-sys 0.61.2")?;
            // This closed dependency edge exposes Windows process/working-set
            // APIs. Source consumers retain their independent impact; it does
            // not change model instrumentation or accelerator execution.
            areas.extend([ChangeArea::Build, ChangeArea::ObservabilityContract]);
        }
        if package == "ferrum-cli" && normalize_launcher_target(&prior, &mut next) {
            areas.insert(ChangeArea::Build);
        }
        if package == "ferrum-native-ops"
            && remove_added_dependency(
                &prior,
                &mut next,
                &["dependencies"],
                "object",
                object_reader_dependency,
            )
        {
            normalize_lock_edge(before, &mut normalized, package, "object")?;
            areas.insert(ChangeArea::Build);
        }
        if package == "ferrum-sampler"
            && remove_added_dependency(
                &prior,
                &mut next,
                &["dependencies"],
                "jsonschema",
                workspace_only,
            )
        {
            normalize_lock_edge(before, &mut normalized, package, "jsonschema")?;
            areas.extend([ChangeArea::Structured, ChangeArea::Tools]);
        }
        if package == "ferrum-kernels" && normalize_nccl_target(&prior, &mut next) {
            // Non-Windows feature unification is preserved; Windows acquires
            // the single-device build without an unavailable NCCL library.
            areas.insert(ChangeArea::Build);
        }
        if !areas.is_empty() {
            paths.insert(path.into(), areas.into_iter().collect());
            normalized.insert(path.into(), next.to_string());
        }
    }
    let validation = validation_dependency_paths(before, &normalized)?;
    if paths.is_empty() {
        return Err("no reviewed product dependency transition".into());
    }
    let all_areas: BTreeSet<_> = paths.values().flatten().copied().collect();
    paths.insert("Cargo.lock".into(), all_areas.into_iter().collect());
    Ok(ProductDependencyRefinement { paths, validation })
}

fn normalize_lock_edge(
    before: &BTreeMap<String, String>,
    normalized: &mut BTreeMap<String, String>,
    package: &str,
    edge: &str,
) -> Result<(), String> {
    let old_lock = parse(before, "Cargo.lock")?;
    let mut new_lock = parse(normalized, "Cargo.lock")?;
    let candidates: Vec<_> = old_lock["package"]
        .as_array_of_tables()
        .ok_or("missing lock packages")?
        .iter()
        .filter(|p| {
            p.get("name").and_then(Item::as_str) == Some(package) && p.get("source").is_none()
        })
        .collect();
    if candidates.len() != 1 {
        return Err(format!("ambiguous local package {package}"));
    }
    let original = candidates[0]
        .get("dependencies")
        .ok_or("missing prior lock edges")?;
    let as_set = |item: &Item| {
        item.as_array().and_then(|a| {
            a.iter()
                .map(Value::as_str)
                .map(|x| x.map(str::to_owned))
                .collect::<Option<BTreeSet<_>>>()
        })
    };
    let mut expected = as_set(original).ok_or("invalid prior lock edges")?;
    if !expected.insert(edge.to_owned()) {
        return Err(format!("lock edge already present: {package}/{edge}"));
    }
    let matching: Vec<_> = new_lock["package"]
        .as_array_of_tables_mut()
        .ok_or("missing lock packages")?
        .iter_mut()
        .filter(|p| {
            p.get("name").and_then(Item::as_str) == Some(package) && p.get("source").is_none()
        })
        .collect();
    if matching.len() != 1 {
        return Err(format!("ambiguous next local package {package}"));
    }
    let next = matching.into_iter().next().unwrap();
    if next.get("dependencies").and_then(as_set) != Some(expected) {
        return Err(format!("unreviewed local lock edges for {package}"));
    }
    next.insert("dependencies", original.clone());
    normalized.insert("Cargo.lock".into(), new_lock.to_string());
    Ok(())
}

fn normalize_training_lock(
    before: &BTreeMap<String, String>,
    normalized: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let old_lock = parse(before, "Cargo.lock")?;
    let mut new_lock = parse(normalized, "Cargo.lock")?;
    let find = |document: &DocumentMut| -> Result<toml_edit::Table, String> {
        let matching: Vec<_> = document["package"]
            .as_array_of_tables()
            .ok_or("lock packages missing")?
            .iter()
            .filter(|p| p.get("name").and_then(Item::as_str) == Some("esaxx-rs"))
            .cloned()
            .collect();
        if matching.len() != 1 {
            return Err("training dependency identity is ambiguous".into());
        }
        Ok(matching[0].clone())
    };
    let previous = find(&old_lock)?;
    let mut current = find(&new_lock)?;
    let dependencies = previous.get("dependencies").and_then(Item::as_array);
    if dependencies
        .is_none_or(|items| items.iter().map(Value::as_str).collect::<Vec<_>>() != [Some("cc")])
        || current.get("dependencies").is_some()
    {
        return Err("training compiler dependency change differs".into());
    }
    current.insert("dependencies", previous["dependencies"].clone());
    if semantic(&Item::Table(current.clone())) != semantic(&Item::Table(previous)) {
        return Err("training registry identity changed".into());
    }
    for package in new_lock["package"]
        .as_array_of_tables_mut()
        .unwrap()
        .iter_mut()
    {
        if package.get("name").and_then(Item::as_str) == Some("esaxx-rs") {
            *package = current.clone();
        }
    }
    normalized.insert("Cargo.lock".into(), new_lock.to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewed_grammar_change_does_not_hide_other_manifest_or_registry_changes() {
        let mut before = super::super::tests::fixture();
        before
            .get_mut("Cargo.toml")
            .unwrap()
            .push_str("llguidance = { version = '1.7', features = ['ahash'] }\n");
        let mut after = before.clone();
        let root = after.get_mut("Cargo.toml").unwrap();
        *root = root.replace("features = ['ahash']", "features = ['ahash', 'lark']");
        let review = product_dependency_paths(&before, &after).unwrap();
        assert!(review.paths["Cargo.toml"].contains(&ChangeArea::Structured));
        assert!(review.paths["Cargo.lock"].contains(&ChangeArea::Tools));
        assert!(!review
            .paths
            .values()
            .flatten()
            .any(|area| *area == ChangeArea::Kernel));

        let mut registry_change = after.clone();
        let lock = registry_change.get_mut("Cargo.lock").unwrap();
        *lock = lock.replace("serde-checksum", "changed-runtime-checksum");
        assert!(product_dependency_paths(&before, &registry_change).is_err());

        after
            .get_mut("crates/app/Cargo.toml")
            .unwrap()
            .push_str("[features]\nchanged_device_route = []\n");
        assert!(product_dependency_paths(&before, &after).is_err());
    }

    #[test]
    fn windows_api_feature_scope_does_not_accept_gpu_features_or_other_sources() {
        let process: DocumentMut = "dep = { version = '0.61.2', features = ['Win32_System_Threading', 'Win32_System_ProcessStatus'] }".parse().unwrap();
        assert!(process_api_dependency(&process["dep"]));
        for spec in [
            "{ version = '0.61.2', features = ['Win32_Graphics_Direct3D'] }",
            "{ version = '0.61.2', features = ['Win32_System_Threading'], path = '../replacement' }",
            "{ version = '0.62', features = ['Win32_System_Threading'] }",
        ] {
            let document: DocumentMut = format!("dep = {spec}").parse().unwrap();
            assert!(!process_api_dependency(&document["dep"]));
        }
    }

    #[test]
    fn target_relocation_keeps_non_windows_nccl_and_other_cuda_features() {
        let before: DocumentMut = r#"
[dependencies]
cudarc = { version = '0.19', optional = true, default-features = false, features = ['std', 'driver'] }
[features]
cuda = ['dep:cudarc', 'marlin', 'cudarc?/nccl']
"#.parse().unwrap();
        let mut after: DocumentMut = r#"
[dependencies]
cudarc = { version = '0.19', optional = true, default-features = false, features = ['std', 'driver'] }
[target.'cfg(not(target_os = "windows"))'.dependencies]
cudarc = { version = '0.19', optional = true, default-features = false, features = ['nccl'] }
[features]
cuda = ['dep:cudarc', 'marlin']
"#.parse().unwrap();
        let mut missing_marlin = after.clone();
        missing_marlin["features"]["cuda"] =
            toml_edit::value(toml_edit::Array::from_iter(["dep:cudarc"]));
        assert!(!normalize_nccl_target(&before, &mut missing_marlin));
        assert!(normalize_nccl_target(&before, &mut after));
        assert_eq!(semantic(before.as_item()), semantic(after.as_item()));
    }

    #[test]
    fn grammar_feature_refinement_keeps_versions_and_unknown_features_visible() {
        let before: DocumentMut =
            "[workspace.dependencies]\nllguidance = { version = '1.7', features = ['ahash'] }"
                .parse()
                .unwrap();
        let mut after: DocumentMut = "[workspace.dependencies]\nllguidance = { version = '1.7', features = ['ahash', 'lark'] }".parse().unwrap();
        assert!(normalize_feature_addition(
            &before,
            &mut after,
            "llguidance",
            "lark"
        ));
        assert_eq!(semantic(after.as_item()), semantic(before.as_item()));
        for different in [
            "{ version = '2', features = ['ahash', 'lark'] }",
            "{ version = '1.7', features = ['ahash', 'lark', 'other'] }",
            "{ version = '1.7', features = ['lark'] }",
        ] {
            let mut after: DocumentMut =
                format!("[workspace.dependencies]\nllguidance = {different}")
                    .parse()
                    .unwrap();
            let original = semantic(after.as_item());
            assert!(!normalize_feature_addition(
                &before,
                &mut after,
                "llguidance",
                "lark"
            ));
            assert_eq!(semantic(after.as_item()), original);
        }
    }
}
