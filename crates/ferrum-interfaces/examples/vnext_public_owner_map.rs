use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use syn::{Fields, ImplItem, Item, TraitItem, UseTree, Visibility};

const MODULES: [&str; 3] = ["resource", "execution", "event"];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SymbolKey {
    public_path: String,
    kind: String,
    macro_contract: Option<String>,
}

#[derive(Clone, Debug)]
struct Symbol {
    key: SymbolKey,
    owner: String,
    root_name: String,
    is_root: bool,
}

struct LoadedModule {
    module: String,
    baseline_symbols: Vec<Symbol>,
    facade: String,
    reexports: Reexports,
    current_symbols: Vec<Symbol>,
    baseline_scope: String,
    current_scope: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OwnerMap {
    schema_version: u32,
    baseline_commit: String,
    baseline_scope: Vec<String>,
    current_scope: Vec<String>,
    item_scope: ItemScope,
    migration_manifest: MigrationManifestEvidence,
    modules: Vec<ModuleMap>,
    summary: Summary,
    diagnostics: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ItemScope {
    top_level_public_items: bool,
    public_struct_and_union_fields: bool,
    public_enum_variants_and_fields: bool,
    public_trait_members: bool,
    public_inherent_impl_members: bool,
    supported_public_macro_invocations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ModuleMap {
    module: String,
    facade: String,
    public_glob_reexports: Vec<String>,
    mappings: Vec<ItemMapping>,
    added_items: Vec<AddedItem>,
    summary: ModuleSummary,
}

#[derive(Debug, Serialize)]
struct ItemMapping {
    old_path: String,
    kind: String,
    old_owner: String,
    new_owner: Option<String>,
    public_path_preserved: bool,
    match_count: usize,
    status: &'static str,
    macro_contract: Option<String>,
    migration: Option<AppliedMigration>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MigrationManifest {
    schema_version: u32,
    baseline_commit: String,
    expected_added_items: usize,
    expected_added_items_sha256: String,
    migrations: Vec<IntentionalMigration>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct IntentionalMigration {
    old_path: String,
    old_kind: String,
    replacement_targets: Vec<MigrationTarget>,
    introduced_by_commit: String,
    rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MigrationTarget {
    path: String,
    kind: String,
}

#[derive(Debug, Serialize)]
struct AppliedMigration {
    replacement_targets: Vec<MigrationTarget>,
    introduced_by_commit: String,
    rationale: String,
}

#[derive(Debug, Serialize)]
struct MigrationManifestEvidence {
    path: String,
    sha256: String,
    migration_count: usize,
    expected_added_items: usize,
    expected_added_items_sha256: String,
}

#[derive(Debug, Serialize)]
struct AddedItem {
    public_path: String,
    kind: String,
    owner: String,
    macro_contract: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModuleSummary {
    baseline_items: usize,
    mapped_items: usize,
    migrated_items: usize,
    lost_items: usize,
    ambiguous_items: usize,
    inaccessible_items: usize,
    added_items: usize,
    excluded_non_public_owner_members: usize,
    coverage_percent: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    baseline_items: usize,
    mapped_items: usize,
    migrated_items: usize,
    lost_items: usize,
    ambiguous_items: usize,
    inaccessible_items: usize,
    added_items: usize,
    added_items_sha256: String,
    excluded_non_public_owner_members: usize,
    unsupported_syntax_count: usize,
    coverage_percent: f64,
    pass: bool,
}

#[derive(Default)]
struct Reexports {
    glob_modules: BTreeSet<String>,
    named_items: BTreeSet<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vnext public owner map failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(format!(
            "usage: vnext_public_owner_map <baseline-commit> <baseline-dir> <current-vnext-dir> <migration-manifest> <output-json>; got {} arguments",
            args.len()
        ));
    }

    let baseline_commit = args[0].clone();
    let baseline_dir = PathBuf::from(&args[1]);
    let current_root = PathBuf::from(&args[2]);
    let migration_manifest_path = PathBuf::from(&args[3]);
    let output_path = PathBuf::from(&args[4]);
    let migration_manifest_bytes = fs::read(&migration_manifest_path).map_err(|error| {
        format!(
            "read migration manifest {}: {error}",
            migration_manifest_path.display()
        )
    })?;
    let migration_manifest = serde_json::from_slice::<MigrationManifest>(&migration_manifest_bytes)
        .map_err(|error| {
            format!(
                "parse migration manifest {}: {error}",
                migration_manifest_path.display()
            )
        })?;
    validate_migration_manifest(&migration_manifest, &baseline_commit)?;
    let migration_manifest_sha256 = format!("{:x}", Sha256::digest(&migration_manifest_bytes));
    let mut migrations_by_key = migration_manifest
        .migrations
        .iter()
        .map(|migration| {
            (
                SymbolKey {
                    public_path: migration.old_path.clone(),
                    kind: migration.old_kind.clone(),
                    macro_contract: None,
                },
                migration,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if migrations_by_key.len() != migration_manifest.migrations.len() {
        return Err("migration manifest contains duplicate old path/kind entries".to_string());
    }
    let mut diagnostics = Vec::new();
    let mut module_maps = Vec::new();
    let mut baseline_scope = Vec::new();
    let mut current_scope = Vec::new();

    let mut loaded_modules = Vec::new();
    for module in MODULES {
        loaded_modules.push(load_module(
            module,
            &baseline_dir,
            &current_root,
            &mut diagnostics,
        )?);
    }
    let external_public_roots = collect_external_public_roots(&current_root, &mut diagnostics)?;
    let mut baseline_public_roots = external_public_roots.clone();
    let mut current_public_roots = external_public_roots;
    for loaded in &loaded_modules {
        baseline_public_roots.extend(
            loaded
                .baseline_symbols
                .iter()
                .filter(|symbol| symbol.is_root)
                .map(|symbol| symbol.root_name.clone()),
        );
        current_public_roots.extend(
            loaded
                .current_symbols
                .iter()
                .filter(|symbol| {
                    symbol.is_root && directly_reexported(symbol, &loaded.facade, &loaded.reexports)
                })
                .map(|symbol| symbol.root_name.clone()),
        );
    }

    for mut loaded in loaded_modules {
        baseline_scope.push(loaded.baseline_scope);
        current_scope.extend(loaded.current_scope);
        let baseline_symbol_count = loaded.baseline_symbols.len();
        loaded
            .baseline_symbols
            .retain(|symbol| symbol.is_root || baseline_public_roots.contains(&symbol.root_name));
        let excluded_non_public_owner_members =
            baseline_symbol_count - loaded.baseline_symbols.len();

        let mut current_by_key = BTreeMap::<SymbolKey, Vec<&Symbol>>::new();
        for symbol in &loaded.current_symbols {
            current_by_key
                .entry(symbol.key.clone())
                .or_default()
                .push(symbol);
        }

        let baseline_keys = loaded
            .baseline_symbols
            .iter()
            .map(|symbol| symbol.key.clone())
            .collect::<BTreeSet<_>>();
        let mut mappings = Vec::new();
        for old in &loaded.baseline_symbols {
            let matches = current_by_key
                .get(&old.key)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let accessible = matches
                .iter()
                .filter(|candidate| publicly_accessible(candidate, &current_public_roots))
                .copied()
                .collect::<Vec<_>>();
            let migration = migrations_by_key.remove(&old.key);
            let (new_owner, status, preserved, applied_migration) = match (
                matches.len(),
                accessible.len(),
                migration,
            ) {
                (0, _, Some(migration)) => {
                    let owners = validate_migration_targets(
                        migration,
                        &current_by_key,
                        &current_public_roots,
                    )?;
                    (
                        Some(owners.join(",")),
                        "migrated",
                        false,
                        Some(AppliedMigration {
                            replacement_targets: migration.replacement_targets.clone(),
                            introduced_by_commit: migration.introduced_by_commit.clone(),
                            rationale: migration.rationale.clone(),
                        }),
                    )
                }
                (_, _, Some(_)) => {
                    let message = format!(
                        "migration manifest entry is redundant because the old symbol still maps: {} ({})",
                        old.key.public_path, old.key.kind
                    );
                    return Err(message);
                }
                (0, _, None) => (None, "lost", false, None),
                (1, 0, None) => (Some(matches[0].owner.clone()), "inaccessible", false, None),
                (1, 1, None) => (Some(accessible[0].owner.clone()), "mapped", true, None),
                (_, 1, None) => (Some(accessible[0].owner.clone()), "ambiguous", false, None),
                (_, _, None) => (None, "ambiguous", false, None),
            };
            mappings.push(ItemMapping {
                old_path: old.key.public_path.clone(),
                kind: old.key.kind.clone(),
                old_owner: old.owner.clone(),
                new_owner,
                public_path_preserved: preserved,
                match_count: matches.len(),
                status,
                macro_contract: old.key.macro_contract.clone(),
                migration: applied_migration,
            });
        }
        mappings.sort_by(|left, right| {
            (&left.old_path, &left.kind).cmp(&(&right.old_path, &right.kind))
        });

        let mut added_items = loaded
            .current_symbols
            .iter()
            .filter(|symbol| {
                publicly_accessible(symbol, &current_public_roots)
                    && !baseline_keys.contains(&symbol.key)
            })
            .map(|symbol| AddedItem {
                public_path: symbol.key.public_path.clone(),
                kind: symbol.key.kind.clone(),
                owner: symbol.owner.clone(),
                macro_contract: symbol.key.macro_contract.clone(),
            })
            .collect::<Vec<_>>();
        added_items.sort_by(|left, right| {
            (&left.public_path, &left.kind).cmp(&(&right.public_path, &right.kind))
        });

        let mapped_items = mappings
            .iter()
            .filter(|mapping| mapping.status == "mapped")
            .count();
        let migrated_items = mappings
            .iter()
            .filter(|mapping| mapping.status == "migrated")
            .count();
        let baseline_items = mappings.len();
        let module_summary = ModuleSummary {
            baseline_items,
            mapped_items,
            migrated_items,
            lost_items: count_status(&mappings, "lost"),
            ambiguous_items: count_status(&mappings, "ambiguous"),
            inaccessible_items: count_status(&mappings, "inaccessible"),
            added_items: added_items.len(),
            excluded_non_public_owner_members,
            coverage_percent: percentage(mapped_items + migrated_items, baseline_items),
        };
        module_maps.push(ModuleMap {
            module: loaded.module,
            facade: loaded.facade,
            public_glob_reexports: loaded.reexports.glob_modules.into_iter().collect(),
            mappings,
            added_items,
            summary: module_summary,
        });
    }

    baseline_scope.sort();
    current_scope.sort();
    current_scope.dedup();
    diagnostics.sort();
    diagnostics.dedup();
    if !migrations_by_key.is_empty() {
        let unused = migrations_by_key
            .keys()
            .map(|key| format!("{} ({})", key.public_path, key.kind))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "migration manifest entries do not identify baseline symbols: {unused}"
        ));
    }
    let baseline_items = module_maps
        .iter()
        .map(|module| module.summary.baseline_items)
        .sum();
    let mapped_items = module_maps
        .iter()
        .map(|module| module.summary.mapped_items)
        .sum();
    let migrated_items = module_maps
        .iter()
        .map(|module| module.summary.migrated_items)
        .sum();
    let lost_items = module_maps
        .iter()
        .map(|module| module.summary.lost_items)
        .sum();
    let ambiguous_items = module_maps
        .iter()
        .map(|module| module.summary.ambiguous_items)
        .sum();
    let inaccessible_items = module_maps
        .iter()
        .map(|module| module.summary.inaccessible_items)
        .sum();
    let added_items = module_maps
        .iter()
        .map(|module| module.summary.added_items)
        .sum();
    let added_items_sha256 = added_items_sha256(&module_maps);
    let excluded_non_public_owner_members = module_maps
        .iter()
        .map(|module| module.summary.excluded_non_public_owner_members)
        .sum();
    let pass = baseline_items > 0
        && mapped_items + migrated_items == baseline_items
        && migrated_items == migration_manifest.migrations.len()
        && lost_items == 0
        && ambiguous_items == 0
        && inaccessible_items == 0
        && added_items == migration_manifest.expected_added_items
        && added_items_sha256 == migration_manifest.expected_added_items_sha256
        && diagnostics.is_empty();
    let output = OwnerMap {
        schema_version: 1,
        baseline_commit,
        baseline_scope,
        current_scope,
        item_scope: ItemScope {
            top_level_public_items: true,
            public_struct_and_union_fields: true,
            public_enum_variants_and_fields: true,
            public_trait_members: true,
            public_inherent_impl_members: true,
            supported_public_macro_invocations: vec![
                "nonzero_execution_id".to_string(),
                "scoped_resource_admission_request".to_string(),
            ],
        },
        migration_manifest: MigrationManifestEvidence {
            path: migration_manifest_path.display().to_string(),
            sha256: migration_manifest_sha256,
            migration_count: migration_manifest.migrations.len(),
            expected_added_items: migration_manifest.expected_added_items,
            expected_added_items_sha256: migration_manifest.expected_added_items_sha256.clone(),
        },
        modules: module_maps,
        summary: Summary {
            baseline_items,
            mapped_items,
            migrated_items,
            lost_items,
            ambiguous_items,
            inaccessible_items,
            added_items,
            added_items_sha256: added_items_sha256.clone(),
            excluded_non_public_owner_members,
            unsupported_syntax_count: diagnostics.len(),
            coverage_percent: percentage(mapped_items + migrated_items, baseline_items),
            pass,
        },
        diagnostics,
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("serialize owner map: {error}"))?;
    fs::write(&output_path, encoded)
        .map_err(|error| format!("write {}: {error}", output_path.display()))?;

    println!(
        "VNEXT PUBLIC OWNER MAP {}: mapped={}/{} migrated={} lost={} ambiguous={} inaccessible={} added={} added_sha256={} unsupported={} output={}",
        if pass { "PASS" } else { "FAIL" },
        mapped_items,
        baseline_items,
        migrated_items,
        lost_items,
        ambiguous_items,
        inaccessible_items,
        added_items,
        added_items_sha256,
        output.summary.unsupported_syntax_count,
        output_path.display()
    );
    if pass {
        Ok(())
    } else {
        Err("owner map acceptance criteria failed; inspect the emitted JSON".to_string())
    }
}

fn validate_migration_manifest(
    manifest: &MigrationManifest,
    baseline_commit: &str,
) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported migration manifest schema version {}",
            manifest.schema_version
        ));
    }
    if manifest.baseline_commit != baseline_commit {
        return Err(format!(
            "migration manifest baseline {} differs from requested baseline {baseline_commit}",
            manifest.baseline_commit
        ));
    }
    if manifest.expected_added_items == 0
        || manifest.expected_added_items_sha256.len() != 64
        || !manifest
            .expected_added_items_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("migration manifest added-item expectation is invalid".to_string());
    }
    if manifest.migrations.is_empty() {
        return Err("migration manifest contains no migrations".to_string());
    }
    for migration in &manifest.migrations {
        if migration.old_path.is_empty()
            || migration.old_kind.is_empty()
            || migration.replacement_targets.is_empty()
            || migration.introduced_by_commit.len() != 40
            || !migration
                .introduced_by_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || migration.rationale.trim().is_empty()
        {
            return Err(format!(
                "migration manifest entry is incomplete: {} ({})",
                migration.old_path, migration.old_kind
            ));
        }
        let targets = migration
            .replacement_targets
            .iter()
            .map(|target| (&target.path, &target.kind))
            .collect::<BTreeSet<_>>();
        if targets.len() != migration.replacement_targets.len()
            || targets
                .iter()
                .any(|(path, kind)| path.is_empty() || kind.is_empty())
        {
            return Err(format!(
                "migration manifest entry has invalid replacement targets: {} ({})",
                migration.old_path, migration.old_kind
            ));
        }
    }
    Ok(())
}

fn validate_migration_targets(
    migration: &IntentionalMigration,
    current_by_key: &BTreeMap<SymbolKey, Vec<&Symbol>>,
    current_public_roots: &BTreeSet<String>,
) -> Result<Vec<String>, String> {
    let mut owners = BTreeSet::new();
    for target in &migration.replacement_targets {
        let key = SymbolKey {
            public_path: target.path.clone(),
            kind: target.kind.clone(),
            macro_contract: None,
        };
        let matches = current_by_key
            .get(&key)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let accessible = matches
            .iter()
            .filter(|candidate| publicly_accessible(candidate, current_public_roots))
            .copied()
            .collect::<Vec<_>>();
        if matches.len() != 1 || accessible.len() != 1 {
            return Err(format!(
                "migration target must resolve to one public symbol: {} ({}) matches={} accessible={}",
                target.path,
                target.kind,
                matches.len(),
                accessible.len()
            ));
        }
        owners.insert(accessible[0].owner.clone());
    }
    Ok(owners.into_iter().collect())
}

fn added_items_sha256(modules: &[ModuleMap]) -> String {
    let mut rows = modules
        .iter()
        .flat_map(|module| {
            module.added_items.iter().map(|item| {
                format!(
                    "{}\t{}\t{}\t{}\t{}\n",
                    module.module,
                    item.public_path,
                    item.kind,
                    item.owner,
                    item.macro_contract.as_deref().unwrap_or("")
                )
            })
        })
        .collect::<Vec<_>>();
    rows.sort();
    format!("{:x}", Sha256::digest(rows.concat().as_bytes()))
}

fn load_module(
    module: &str,
    baseline_dir: &Path,
    current_root: &Path,
    diagnostics: &mut Vec<String>,
) -> Result<LoadedModule, String> {
    let baseline_path = baseline_dir.join(format!("{module}.rs"));
    let facade_path = current_root.join(format!("{module}.rs"));
    let module_dir = current_root.join(module);
    require_file(&baseline_path)?;
    require_file(&facade_path)?;
    require_directory(&module_dir)?;

    let baseline_scope = format!("baseline/{module}.rs");
    let baseline_file = parse_file(&baseline_path)?;
    let mut baseline_symbols = Vec::new();
    collect_symbols(
        module,
        &baseline_scope,
        &baseline_file.items,
        &mut baseline_symbols,
        diagnostics,
    );

    let facade = relative_owner(current_root, &facade_path)?;
    let facade_file = parse_file(&facade_path)?;
    let reexports = collect_reexports(&facade_file.items, &facade, diagnostics);
    let mut current_symbols = Vec::new();
    collect_symbols(
        module,
        &facade,
        &facade_file.items,
        &mut current_symbols,
        diagnostics,
    );
    let mut current_scope = vec![facade.clone()];

    let mut owner_paths = fs::read_dir(&module_dir)
        .map_err(|error| format!("read {}: {error}", module_dir.display()))?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read {} entry: {error}", module_dir.display()))?;
    owner_paths.sort();
    for owner_path in owner_paths {
        if owner_path.extension().and_then(|value| value.to_str()) != Some("rs")
            || is_test_support(&owner_path)
        {
            continue;
        }
        let owner = relative_owner(current_root, &owner_path)?;
        let syntax = parse_file(&owner_path)?;
        collect_symbols(
            module,
            &owner,
            &syntax.items,
            &mut current_symbols,
            diagnostics,
        );
        current_scope.push(owner);
    }

    Ok(LoadedModule {
        module: module.to_string(),
        baseline_symbols,
        facade,
        reexports,
        current_symbols,
        baseline_scope,
        current_scope,
    })
}

fn collect_external_public_roots(
    current_root: &Path,
    _diagnostics: &mut Vec<String>,
) -> Result<BTreeSet<String>, String> {
    let mut paths = fs::read_dir(current_root)
        .map_err(|error| format!("read {}: {error}", current_root.display()))?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read {} entry: {error}", current_root.display()))?;
    paths.sort();
    let split_modules = MODULES.into_iter().collect::<BTreeSet<_>>();
    let mut roots = BTreeSet::new();
    for path in paths {
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        if path.extension().and_then(|value| value.to_str()) != Some("rs")
            || stem == "mod"
            || split_modules.contains(stem)
        {
            continue;
        }
        let owner = relative_owner(current_root, &path)?;
        let syntax = parse_file(&path)?;
        for item in &syntax.items {
            if let Some(name) = explicit_public_root_name(item) {
                roots.insert(name);
                continue;
            }
            let Item::Macro(item_macro) = item else {
                continue;
            };
            if item_macro.ident.is_some()
                || item_macro
                    .mac
                    .path
                    .segments
                    .last()
                    .is_none_or(|segment| segment.ident != "stable_identity")
            {
                continue;
            }
            let arguments = syn::parse2::<MacroArguments>(item_macro.mac.tokens.clone())
                .map_err(|error| format!("{owner}: parse stable_identity arguments: {error}"))?;
            roots.insert(arguments.first_ident()?.to_string());
        }
    }
    Ok(roots)
}

fn explicit_public_root_name(item: &Item) -> Option<String> {
    match item {
        Item::Const(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Enum(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Fn(value) if is_public(&value.vis) => Some(value.sig.ident.to_string()),
        Item::Mod(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Static(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Struct(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Trait(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::TraitAlias(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Type(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        Item::Union(value) if is_public(&value.vis) => Some(value.ident.to_string()),
        _ => None,
    }
}

fn collect_symbols(
    module: &str,
    owner: &str,
    items: &[Item],
    symbols: &mut Vec<Symbol>,
    diagnostics: &mut Vec<String>,
) {
    for item in items {
        match item {
            Item::Const(value) if is_public(&value.vis) => push_symbol(
                symbols,
                module,
                owner,
                value.ident.to_string(),
                "const",
                None,
            ),
            Item::Enum(value) if is_public(&value.vis) => {
                let name = value.ident.to_string();
                push_symbol(symbols, module, owner, name.clone(), "enum", None);
                for variant in &value.variants {
                    let variant_name = format!("{name}::{}", variant.ident);
                    push_symbol(
                        symbols,
                        module,
                        owner,
                        variant_name.clone(),
                        "enum_variant",
                        None,
                    );
                    collect_fields(symbols, module, owner, &variant_name, &variant.fields, true);
                }
            }
            Item::Fn(value) if is_public(&value.vis) => push_symbol(
                symbols,
                module,
                owner,
                value.sig.ident.to_string(),
                "function",
                None,
            ),
            Item::Impl(value) if value.trait_.is_none() => {
                let Some(owner_type) = type_name(&value.self_ty) else {
                    diagnostics.push(format!("{owner}: unsupported inherent impl self type"));
                    continue;
                };
                for member in &value.items {
                    match member {
                        ImplItem::Const(item) if is_public(&item.vis) => push_symbol(
                            symbols,
                            module,
                            owner,
                            format!("{owner_type}::{}", item.ident),
                            "inherent_const",
                            None,
                        ),
                        ImplItem::Fn(item) if is_public(&item.vis) => push_symbol(
                            symbols,
                            module,
                            owner,
                            format!("{owner_type}::{}", item.sig.ident),
                            "inherent_method",
                            None,
                        ),
                        ImplItem::Type(item) if is_public(&item.vis) => push_symbol(
                            symbols,
                            module,
                            owner,
                            format!("{owner_type}::{}", item.ident),
                            "inherent_type",
                            None,
                        ),
                        _ => {}
                    }
                }
            }
            Item::Macro(value) if value.ident.is_none() => {
                let macro_name = value
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|segment| segment.ident.to_string())
                    .unwrap_or_default();
                if matches!(
                    macro_name.as_str(),
                    "nonzero_execution_id" | "scoped_resource_admission_request"
                ) {
                    match syn::parse2::<MacroArguments>(value.mac.tokens.clone()) {
                        Ok(arguments) => match arguments.first_ident() {
                            Ok(ident) => push_symbol(
                                symbols,
                                module,
                                owner,
                                ident.to_string(),
                                "macro_generated_item",
                                Some(format!("{}!({})", macro_name, value.mac.tokens.to_string())),
                            ),
                            Err(error) => diagnostics.push(format!(
                                "{owner}: cannot identify generated item for {macro_name}: {error}"
                            )),
                        },
                        Err(error) => diagnostics.push(format!(
                            "{owner}: cannot parse first identifier for {macro_name}: {error}"
                        )),
                    }
                } else {
                    diagnostics.push(format!(
                        "{owner}: unsupported top-level macro invocation {macro_name}"
                    ));
                }
            }
            Item::Mod(value) if is_public(&value.vis) => {
                let name = value.ident.to_string();
                push_symbol(symbols, module, owner, name.clone(), "module", None);
                if let Some((_, nested)) = &value.content {
                    collect_symbols(module, owner, nested, symbols, diagnostics);
                } else {
                    diagnostics.push(format!(
                        "{owner}: external public module {name} is unsupported"
                    ));
                }
            }
            Item::Static(value) if is_public(&value.vis) => push_symbol(
                symbols,
                module,
                owner,
                value.ident.to_string(),
                "static",
                None,
            ),
            Item::Struct(value) if is_public(&value.vis) => {
                let name = value.ident.to_string();
                push_symbol(symbols, module, owner, name.clone(), "struct", None);
                collect_fields(symbols, module, owner, &name, &value.fields, false);
            }
            Item::Trait(value) if is_public(&value.vis) => {
                let name = value.ident.to_string();
                push_symbol(symbols, module, owner, name.clone(), "trait", None);
                for member in &value.items {
                    match member {
                        TraitItem::Const(item) => push_symbol(
                            symbols,
                            module,
                            owner,
                            format!("{name}::{}", item.ident),
                            "trait_const",
                            None,
                        ),
                        TraitItem::Fn(item) => push_symbol(
                            symbols,
                            module,
                            owner,
                            format!("{name}::{}", item.sig.ident),
                            "trait_method",
                            None,
                        ),
                        TraitItem::Type(item) => push_symbol(
                            symbols,
                            module,
                            owner,
                            format!("{name}::{}", item.ident),
                            "trait_type",
                            None,
                        ),
                        TraitItem::Macro(item) => diagnostics.push(format!(
                            "{owner}: public trait {name} contains unsupported macro {}",
                            item.mac
                                .path
                                .segments
                                .last()
                                .map(|value| value.ident.to_string())
                                .unwrap_or_default()
                        )),
                        _ => {}
                    }
                }
            }
            Item::TraitAlias(value) if is_public(&value.vis) => push_symbol(
                symbols,
                module,
                owner,
                value.ident.to_string(),
                "trait_alias",
                None,
            ),
            Item::Type(value) if is_public(&value.vis) => push_symbol(
                symbols,
                module,
                owner,
                value.ident.to_string(),
                "type_alias",
                None,
            ),
            Item::Union(value) if is_public(&value.vis) => {
                let name = value.ident.to_string();
                push_symbol(symbols, module, owner, name.clone(), "union", None);
                for field in &value.fields.named {
                    if is_public(&field.vis) {
                        if let Some(ident) = &field.ident {
                            push_symbol(
                                symbols,
                                module,
                                owner,
                                format!("{name}::{ident}"),
                                "union_field",
                                None,
                            );
                        }
                    }
                }
            }
            Item::Use(value) if is_public(&value.vis) => {}
            Item::Verbatim(_) => {
                diagnostics.push(format!("{owner}: unsupported top-level verbatim syntax"))
            }
            _ => {}
        }
    }
}

fn collect_fields(
    symbols: &mut Vec<Symbol>,
    module: &str,
    owner: &str,
    parent: &str,
    fields: &Fields,
    enum_fields_are_public: bool,
) {
    for (index, field) in fields.iter().enumerate() {
        if enum_fields_are_public || is_public(&field.vis) {
            let name = field
                .ident
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| index.to_string());
            push_symbol(
                symbols,
                module,
                owner,
                format!("{parent}::{name}"),
                if enum_fields_are_public {
                    "enum_field"
                } else {
                    "struct_field"
                },
                None,
            );
        }
    }
}

fn push_symbol(
    symbols: &mut Vec<Symbol>,
    _module: &str,
    owner: &str,
    relative_path: String,
    kind: &str,
    macro_contract: Option<String>,
) {
    let is_root = !relative_path.contains("::");
    let root_name = relative_path
        .split("::")
        .next()
        .unwrap_or(&relative_path)
        .to_string();
    symbols.push(Symbol {
        key: SymbolKey {
            public_path: format!("ferrum_interfaces::vnext::{relative_path}"),
            kind: kind.to_string(),
            macro_contract,
        },
        owner: owner.to_string(),
        root_name,
        is_root,
    });
}

fn collect_reexports(items: &[Item], owner: &str, diagnostics: &mut Vec<String>) -> Reexports {
    let mut reexports = Reexports::default();
    for item in items {
        let Item::Use(item_use) = item else {
            continue;
        };
        if !is_public(&item_use.vis) {
            continue;
        }
        collect_use_tree(
            &item_use.tree,
            &mut Vec::new(),
            &mut reexports,
            owner,
            diagnostics,
        );
    }
    reexports
}

fn collect_use_tree(
    tree: &UseTree,
    prefix: &mut Vec<String>,
    reexports: &mut Reexports,
    owner: &str,
    diagnostics: &mut Vec<String>,
) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_tree(&path.tree, prefix, reexports, owner, diagnostics);
            prefix.pop();
        }
        UseTree::Name(name) => {
            if prefix.is_empty() {
                reexports.named_items.insert(name.ident.to_string());
            } else {
                reexports.named_items.insert(name.ident.to_string());
            }
        }
        UseTree::Rename(rename) => {
            reexports.named_items.insert(rename.rename.to_string());
        }
        UseTree::Glob(_) => {
            if let Some(module) = prefix.last() {
                reexports.glob_modules.insert(module.clone());
            } else {
                diagnostics.push(format!("{owner}: unsupported root glob reexport"));
            }
        }
        UseTree::Group(group) => {
            for nested in &group.items {
                collect_use_tree(nested, prefix, reexports, owner, diagnostics);
            }
        }
    }
}

fn directly_reexported(symbol: &Symbol, facade: &str, reexports: &Reexports) -> bool {
    if symbol.owner == facade {
        return true;
    }
    let module = Path::new(&symbol.owner)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    reexports.glob_modules.contains(module) || reexports.named_items.contains(&symbol.root_name)
}

fn publicly_accessible(symbol: &Symbol, publicly_reexported_roots: &BTreeSet<String>) -> bool {
    publicly_reexported_roots.contains(&symbol.root_name)
}

fn type_name(value: &syn::Type) -> Option<String> {
    match value {
        syn::Type::Path(path) if path.qself.is_none() => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        syn::Type::Group(group) => type_name(&group.elem),
        syn::Type::Paren(paren) => type_name(&paren.elem),
        _ => None,
    }
}

struct MacroArguments(syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>);

impl MacroArguments {
    fn first_ident(&self) -> Result<&syn::Ident, String> {
        let Some(syn::Expr::Path(path)) = self.0.first() else {
            return Err("supported public item macro must start with an identifier".to_string());
        };
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return Err("supported public item macro must start with one identifier".to_string());
        }
        Ok(&path.path.segments[0].ident)
    }
}

impl syn::parse::Parse for MacroArguments {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self(syn::punctuated::Punctuated::<
            syn::Expr,
            syn::Token![,],
        >::parse_terminated(input)?))
    }
}

fn is_public(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

fn is_test_support(path: &Path) -> bool {
    path.file_stem()
        .and_then(|value| value.to_str())
        .is_some_and(|stem| stem == "tests" || stem.ends_with("_tests"))
}

fn parse_file(path: &Path) -> Result<syn::File, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    syn::parse_file(&source).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn relative_owner(root: &Path, path: &Path) -> Result<String, String> {
    path.strip_prefix(root)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .map_err(|error| {
            format!(
                "{} is not under {}: {error}",
                path.display(),
                root.display()
            )
        })
}

fn require_file(path: &Path) -> Result<(), String> {
    if path.is_file() && !path.is_symlink() {
        Ok(())
    } else {
        Err(format!(
            "required regular file is missing: {}",
            path.display()
        ))
    }
}

fn require_directory(path: &Path) -> Result<(), String> {
    if path.is_dir() && !path.is_symlink() {
        Ok(())
    } else {
        Err(format!("required directory is missing: {}", path.display()))
    }
}

fn count_status(mappings: &[ItemMapping], status: &str) -> usize {
    mappings
        .iter()
        .filter(|mapping| mapping.status == status)
        .count()
}

fn percentage(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 * 100.0 / denominator as f64
    }
}
