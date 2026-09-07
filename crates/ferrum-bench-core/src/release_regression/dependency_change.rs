//! Narrow development/tool dependency changes from complete immutable Cargo inputs.
//! Unknown runtime, feature, build, workspace or existing registry changes retain
//! conservative impact. This is dependency classification, never test evidence.
use crate::release_candidate::version::{plan_version_update, MemberManifest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use toml_edit::{DocumentMut, Item, TableLike, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyRefinement {
    pub paths: Vec<String>,
    pub coordinated_version: bool,
    /// These reviewed validation dependencies are ordinary Cargo dependencies;
    /// do not claim that adding their edges leaves the entire Cargo graph equal.
    pub validation_runtime_dependencies: Vec<String>,
    /// Explicit private workspace tools whose addition was proven isolated.
    #[serde(default)]
    pub private_tool_members: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Semantic {
    String(String),
    Integer(i64),
    Float(String),
    Bool(bool),
    Datetime(String),
    Array(Vec<Semantic>),
    Table(BTreeMap<String, Semantic>),
    None,
}
fn semantic_value(value: &Value) -> Semantic {
    match value {
        Value::String(v) => Semantic::String(v.value().clone()),
        Value::Integer(v) => Semantic::Integer(*v.value()),
        Value::Float(v) => Semantic::Float(v.value().to_string()),
        Value::Boolean(v) => Semantic::Bool(*v.value()),
        Value::Datetime(v) => Semantic::Datetime(v.value().to_string()),
        Value::Array(v) => Semantic::Array(v.iter().map(semantic_value).collect()),
        Value::InlineTable(v) => Semantic::Table(
            v.iter()
                .map(|(key, v)| (key.into(), semantic_value(v)))
                .collect(),
        ),
    }
}
fn semantic(item: &Item) -> Semantic {
    match item {
        Item::None => Semantic::None,
        Item::Value(v) => semantic_value(v),
        Item::Table(table) => Semantic::Table(
            table
                .iter()
                .map(|(key, item)| (key.into(), semantic(item)))
                .collect(),
        ),
        Item::ArrayOfTables(tables) => Semantic::Array(
            tables
                .iter()
                .map(|table| {
                    Semantic::Table(
                        table
                            .iter()
                            .map(|(key, item)| (key.into(), semantic(item)))
                            .collect(),
                    )
                })
                .collect(),
        ),
    }
}
fn parse(files: &BTreeMap<String, String>, path: &str) -> Result<DocumentMut, String> {
    files
        .get(path)
        .ok_or_else(|| format!("missing Cargo input {path}"))?
        .parse()
        .map_err(|error| format!("invalid {path}: {error}"))
}
fn names(
    root: &DocumentMut,
    files: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, String> {
    let workspace = root
        .get("workspace")
        .and_then(Item::as_table_like)
        .ok_or("missing workspace")?;
    if !matches!(
        workspace.get("resolver").and_then(Item::as_str),
        Some("2" | "3")
    ) {
        return Err("dev dependency refinement requires resolver 2 or 3".into());
    }
    if workspace
        .get("exclude")
        .is_some_and(|item| !item.as_array().is_some_and(|array| array.is_empty()))
    {
        return Err("workspace exclusions are not refined".into());
    }
    let members = workspace
        .get("members")
        .and_then(Item::as_array)
        .ok_or("workspace members must be explicit")?;
    let mut paths = Vec::new();
    for member in members {
        let member = member.as_str().ok_or("workspace member must be a path")?;
        if member.contains(['*', '?', '[', ']', '{', '}', ':', '\\'])
            || member
                .split('/')
                .any(|part| matches!(part, "" | "." | ".."))
        {
            return Err("nonliteral workspace member is not refined".into());
        }
        paths.push(format!("{member}/Cargo.toml"));
    }
    if root.get("package").is_some() {
        paths.push("Cargo.toml".into());
    }
    let mut result = BTreeMap::new();
    for path in paths {
        let document = parse(files, &path)?;
        let name = document
            .get("package")
            .and_then(|item| item.get("name"))
            .and_then(Item::as_str)
            .ok_or("member package.name must be explicit")?;
        if result.insert(name.into(), path).is_some() {
            return Err("duplicate workspace package".into());
        }
    }
    Ok(result)
}
fn package_name(key: &str, spec: &Item, root: &DocumentMut) -> Result<String, String> {
    if spec.get("workspace").and_then(Item::as_bool) == Some(true) {
        let inherited = root
            .get("workspace")
            .and_then(|item| item.get("dependencies"))
            .and_then(|item| item.get(key))
            .ok_or("missing workspace dependency")?;
        return Ok(inherited
            .get("package")
            .and_then(Item::as_str)
            .unwrap_or(key)
            .into());
    }
    Ok(spec
        .get("package")
        .and_then(Item::as_str)
        .unwrap_or(key)
        .into())
}
fn remove_dev(
    table: &mut dyn TableLike,
    root: &DocumentMut,
    names: &mut BTreeSet<String>,
) -> Result<(), String> {
    if let Some(dependencies) = table.remove("dev-dependencies") {
        let dependencies = dependencies
            .as_table_like()
            .ok_or("dev-dependencies must be a table")?;
        for (key, spec) in dependencies.iter() {
            names.insert(package_name(key, spec, root)?);
        }
    }
    Ok(())
}
fn strip_dev(document: &mut DocumentMut, root: &DocumentMut) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    remove_dev(document.as_table_mut(), root, &mut names)?;
    if let Some(targets) = document.get_mut("target") {
        let targets = targets
            .as_table_like_mut()
            .ok_or("target must be a table")?;
        let keys: Vec<_> = targets.iter().map(|(key, _)| key.to_owned()).collect();
        for key in keys {
            let target = targets
                .get_mut(&key)
                .and_then(Item::as_table_like_mut)
                .ok_or("target condition must be a table")?;
            remove_dev(target, root, &mut names)?;
            if target.is_empty() {
                targets.remove(&key);
            }
        }
        if targets.is_empty() {
            document.as_table_mut().remove("target");
        }
    }
    Ok(names)
}

fn runtime_names(document: &DocumentMut, root: &DocumentMut) -> Result<BTreeSet<String>, String> {
    let mut result = BTreeSet::new();
    let mut collect = |table: &dyn TableLike| -> Result<(), String> {
        for kind in ["dependencies", "build-dependencies"] {
            if let Some(dependencies) = table.get(kind) {
                let dependencies = dependencies
                    .as_table_like()
                    .ok_or("dependencies must be tables")?;
                for (name, spec) in dependencies.iter() {
                    result.insert(package_name(name, spec, root)?);
                }
            }
        }
        Ok(())
    };
    collect(document.as_table())?;
    if let Some(targets) = document.get("target") {
        for (_, target) in targets
            .as_table_like()
            .ok_or("target must be a table")?
            .iter()
        {
            collect(
                target
                    .as_table_like()
                    .ok_or("target condition must be a table")?,
            )?;
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Package {
    name: String,
    version: String,
    source: Option<String>,
}
struct Locked {
    body: Semantic,
    dependencies: BTreeSet<String>,
}
fn locks(document: &DocumentMut) -> Result<BTreeMap<Package, Locked>, String> {
    let records = document
        .get("package")
        .and_then(Item::as_array_of_tables)
        .ok_or("lock is missing packages")?;
    let mut result = BTreeMap::new();
    for record in records {
        let field = |name| {
            record
                .get(name)
                .and_then(Item::as_str)
                .ok_or("invalid lock package identity")
        };
        let package = Package {
            name: field("name")?.into(),
            version: field("version")?.into(),
            source: record
                .get("source")
                .and_then(Item::as_str)
                .map(str::to_owned),
        };
        let dependencies = match record.get("dependencies") {
            None => BTreeSet::new(),
            Some(item) => item
                .as_array()
                .ok_or("lock dependencies must be an array")?
                .iter()
                .map(|item| {
                    item.as_str()
                        .map(str::to_owned)
                        .ok_or("invalid lock dependency")
                })
                .collect::<Result<_, _>>()?,
        };
        let mut body = record.clone();
        body.remove("dependencies");
        if result
            .insert(
                package,
                Locked {
                    body: semantic(&Item::Table(body)),
                    dependencies,
                },
            )
            .is_some()
        {
            return Err("duplicate lock identity".into());
        }
    }
    Ok(result)
}
fn resolve<'a>(
    reference: &str,
    records: &'a BTreeMap<Package, Locked>,
) -> Result<&'a Package, String> {
    let pieces: Vec<_> = reference.split_whitespace().collect();
    let name = pieces.first().ok_or("empty lock dependency")?;
    let candidates: Vec<_> = records
        .keys()
        .filter(|package| {
            package.name == *name
                && pieces
                    .get(1)
                    .is_none_or(|version| package.version == *version)
                && pieces.get(2).is_none_or(|source| {
                    Some(source.trim_matches(['(', ')'])) == package.source.as_deref()
                })
        })
        .collect();
    if pieces.len() > 3 || candidates.len() != 1 {
        return Err(format!("ambiguous lock dependency {reference}"));
    }
    Ok(candidates[0])
}
fn closure(
    seeds: BTreeSet<Package>,
    records: &BTreeMap<Package, Locked>,
) -> Result<BTreeSet<Package>, String> {
    let mut todo: Vec<_> = seeds.into_iter().collect();
    let mut seen = BTreeSet::new();
    while let Some(package) = todo.pop() {
        if !seen.insert(package.clone()) {
            continue;
        }
        for dependency in &records[&package].dependencies {
            todo.push(resolve(dependency, records)?.clone());
        }
    }
    Ok(seen)
}

const DEVTOOLS: &str = "ferrum-devtools";
const DEVTOOLS_MANIFEST: &str = "crates/ferrum-devtools/Cargo.toml";

/// Only this reviewed private executable crate may be added without declaring
/// new product architecture reach. No build hook, library, feature table or
/// alternative configuration is silently accepted under its private name.
fn private_tool_dependencies(
    files: &BTreeMap<String, String>,
    root: &DocumentMut,
    members: &BTreeMap<String, String>,
) -> Result<BTreeSet<String>, String> {
    if members.get(DEVTOOLS).map(String::as_str) != Some(DEVTOOLS_MANIFEST) {
        return Err("private devtools must use its explicit workspace member path".into());
    }
    let document = parse(files, DEVTOOLS_MANIFEST)?;
    if document
        .as_table()
        .iter()
        .any(|(key, _)| !matches!(key, "package" | "dependencies" | "dev-dependencies" | "bin"))
    {
        return Err("unreviewed private devtools manifest configuration".into());
    }
    let package = document
        .get("package")
        .and_then(Item::as_table_like)
        .ok_or("missing private devtools package")?;
    if package.get("name").and_then(Item::as_str) != Some(DEVTOOLS)
        || package.get("publish").and_then(Item::as_bool) != Some(false)
        || package.iter().any(|(key, _)| {
            !matches!(
                key,
                "name"
                    | "version"
                    | "edition"
                    | "publish"
                    | "license"
                    | "description"
                    | "authors"
                    | "repository"
                    | "homepage"
                    | "rust-version"
            )
        })
    {
        return Err(
            "private devtools must remain publish=false without new build/package configuration"
                .into(),
        );
    }
    let binaries = document
        .get("bin")
        .and_then(Item::as_array_of_tables)
        .ok_or("private devtools must declare its reviewed executable targets")?;
    let mut seen = BTreeSet::new();
    for binary in binaries {
        let name = binary
            .get("name")
            .and_then(Item::as_str)
            .ok_or("private tool bin needs explicit name")?;
        let path = binary
            .get("path")
            .and_then(Item::as_str)
            .ok_or("private tool bin needs explicit path")?;
        if !matches!(
            name,
            "release_delivery" | "contract_checks" | "model_regression"
        ) || path != format!("src/bin/{name}.rs")
            || !seen.insert(name)
            || binary
                .iter()
                .any(|(key, _)| !matches!(key, "name" | "path"))
        {
            return Err("unreviewed private devtools executable target".into());
        }
    }
    if !seen.contains("release_delivery") || !seen.contains("contract_checks") {
        return Err("private devtools must retain release_delivery and contract_checks".into());
    }
    let mut dependencies = BTreeSet::new();
    for kind in ["dependencies", "dev-dependencies"] {
        if let Some(table) = document.get(kind) {
            for (name, spec) in table
                .as_table_like()
                .ok_or("private tool dependencies must be tables")?
                .iter()
            {
                let reviewed = spec.as_str().is_some()
                    || spec.as_table_like().is_some_and(|table| {
                        table.iter().all(|(key, value)| match key {
                            "workspace" => value.as_bool() == Some(true),
                            "version" | "path" | "package" => value.as_str().is_some(),
                            "default-features" => value.as_bool().is_some(),
                            "features" => value.as_array().is_some_and(|values| {
                                values.iter().all(|value| value.as_str().is_some())
                            }),
                            _ => false,
                        }) && (table.get("workspace").and_then(Item::as_bool) == Some(true)
                            || table.get("version").and_then(Item::as_str).is_some()
                            || table.get("path").and_then(Item::as_str).is_some())
                    });
                if !reviewed {
                    return Err(format!("unreviewed private tool dependency {name}"));
                }
                dependencies.insert(package_name(name, spec, root)?);
            }
        }
    }
    for (name, path) in members.iter().filter(|(name, _)| name.as_str() != DEVTOOLS) {
        let mut peer = parse(files, path)?;
        let mut edges = runtime_names(&peer, root)?;
        edges.extend(strip_dev(&mut peer, root)?);
        if edges.contains(DEVTOOLS) {
            return Err(format!(
                "existing member {name} depends on private devtools"
            ));
        }
    }
    Ok(dependencies)
}

/// The runner migration changes private executable selection and two dependency
/// declarations. Normalize only that addition; the caller still compares every
/// other manifest value and all existing locked identities and registry edges.
fn normalize_model_runner_migration(
    prior: &DocumentMut,
    next: &mut DocumentMut,
) -> Result<BTreeSet<String>, String> {
    let runner_index = |document: &DocumentMut| {
        document
            .get("bin")
            .and_then(Item::as_array_of_tables)
            .and_then(|bins| {
                bins.iter().position(|bin| {
                    bin.get("name").and_then(Item::as_str) == Some("model_regression")
                })
            })
    };
    let Some(index) = runner_index(next).filter(|_| runner_index(prior).is_none()) else {
        return Ok(BTreeSet::new());
    };
    // private_tool_dependencies already checked the full target declaration,
    // required existing tools, publish=false and absence of reverse consumers.
    next["bin"]
        .as_array_of_tables_mut()
        .expect("checked bin table")
        .remove(index);
    let mut edges = BTreeSet::new();
    for name in ["anyhow", "ferrum-types"] {
        if prior
            .get("dependencies")
            .and_then(|v| v.get(name))
            .is_some()
        {
            continue;
        }
        let Some(spec) = next.get("dependencies").and_then(|v| v.get(name)) else {
            continue;
        };
        let reviewed = if name == "anyhow" {
            spec.as_table_like().is_some_and(|table| {
                table.len() == 1 && table.get("workspace").and_then(Item::as_bool) == Some(true)
            })
        } else {
            prior
                .get("dev-dependencies")
                .and_then(|v| v.get(name))
                .is_some_and(|old| semantic(old) == semantic(spec))
        };
        if !reviewed {
            return Err(format!(
                "unreviewed model runner dependency migration {name}"
            ));
        }
        next.get_mut("dependencies")
            .and_then(Item::as_table_like_mut)
            .expect("checked dependencies")
            .remove(name);
        edges.insert(name.to_owned());
    }
    Ok(edges)
}

fn remove_private_member(root: &mut DocumentMut) -> Result<(), String> {
    let members = root
        .get_mut("workspace")
        .and_then(|item| item.get_mut("members"))
        .and_then(Item::as_array_mut)
        .ok_or("workspace members must be explicit")?;
    let indices: Vec<_> = members
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            (value.as_str() == Some("crates/ferrum-devtools")).then_some(index)
        })
        .collect();
    if indices.len() != 1 {
        return Err("private devtools member must occur once".into());
    }
    members.remove(indices[0]);
    Ok(())
}

const TESTKIT: &str = "ferrum-testkit";
const BENCH_CORE: &str = "ferrum-bench-core";

fn workspace_only(spec: &Item) -> bool {
    spec.as_table_like().is_some_and(|table| {
        table.len() == 1 && table.get("workspace").and_then(Item::as_bool) == Some(true)
    })
}
fn isolated_testkit(
    files: &BTreeMap<String, String>,
    root: &DocumentMut,
    members: &BTreeMap<String, String>,
) -> Result<(), String> {
    for (name, path) in members.iter().filter(|(name, _)| name.as_str() != TESTKIT) {
        if runtime_names(&parse(files, path)?, root)?.contains(TESTKIT) {
            return Err(format!(
                "member {name} has a runtime/build dependency on testkit"
            ));
        }
    }
    Ok(())
}
fn reviewed_local_bench_edge(
    spec: &Item,
    root: &DocumentMut,
    members: &BTreeMap<String, String>,
) -> bool {
    if !workspace_only(spec) {
        return false;
    }
    let inherited = root
        .get("workspace")
        .and_then(|v| v.get("dependencies"))
        .and_then(|v| v.get(BENCH_CORE))
        .and_then(Item::as_table_like);
    inherited.is_some_and(|table| {
        table.len() == 2
            && table.get("version").and_then(Item::as_str).is_some()
            && table.get("path").and_then(Item::as_str)
                == members
                    .get(BENCH_CORE)
                    .and_then(|path| path.strip_suffix("/Cargo.toml"))
    })
}
fn reviewed_zip_feature(spec: &Item) -> bool {
    spec.as_table_like().is_some_and(|table| {
        table.len() == 3
            && table.get("version").and_then(Item::as_str) == Some("7.2.0")
            && table.get("default-features").and_then(Item::as_bool) == Some(false)
            && table
                .get("features")
                .and_then(Item::as_array)
                .is_some_and(|features| {
                    features.len() == 1
                        && features.get(0).and_then(Value::as_str) == Some("deflate-flate2")
                })
    })
}

fn reviewed_private_addition(dependency: &str, spec: &Item) -> bool {
    match dependency {
        "chrono" => workspace_only(spec),
        "zip" => reviewed_zip_feature(spec),
        "flate2" => spec.as_str() == Some("1"),
        _ => false,
    }
}

// The source proof uses a full Rust AST and its visitor. No other parser
// feature, dependency source or default-feature override is assumed harmless.
fn reviewed_syntax_features(spec: &Item) -> Option<bool> {
    let table = spec.as_table_like()?;
    if table.len() != 2 || table.get("version")?.as_str().is_none() {
        return None;
    }
    let features = table.get("features")?.as_array()?;
    let names = features
        .iter()
        .map(Value::as_str)
        .collect::<Option<BTreeSet<_>>>()?;
    if names.len() != features.len() {
        return None;
    }
    if names == BTreeSet::from(["full"]) {
        Some(false)
    } else if names == BTreeSet::from(["full", "visit-mut"]) {
        Some(true)
    } else {
        None
    }
}

/// Complete before/after root/member manifests and Cargo.lock are required. The
/// reviewed ordinary additions are the release parser tools, isolated testkit
/// numerical sharing, and private artifact readers. Source diffs retain their
/// own impact. The syntax visitor and isolated ZIP/flate2 feature edge are the
/// only reviewed extensions; other shared flags and resolution stay conservative.
pub fn validation_dependency_paths(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> Result<DependencyRefinement, String> {
    let added_tool =
        !before.contains_key(DEVTOOLS_MANIFEST) && after.contains_key(DEVTOOLS_MANIFEST);
    if !before.keys().eq(after
        .keys()
        .filter(|path| !(added_tool && path.as_str() == DEVTOOLS_MANIFEST)))
    {
        return Err("Cargo inputs were added or removed outside the private tool member".into());
    }
    let root = parse(before, "Cargo.toml")?;
    let next_root = parse(after, "Cargo.toml")?;
    let members = names(&root, before)?;
    let next_members = names(&next_root, after)?;
    let mut comparable_members = next_members.clone();
    if added_tool {
        comparable_members.remove(DEVTOOLS);
    }
    if members != comparable_members {
        return Err("workspace membership changed outside the private tool member".into());
    }
    let tool_dependencies = if next_members.contains_key(DEVTOOLS) {
        Some(private_tool_dependencies(after, &next_root, &next_members)?)
    } else {
        None
    };
    let version = |document: &DocumentMut| {
        document
            .get("workspace")
            .and_then(|item| item.get("package"))
            .and_then(|item| item.get("version"))
            .and_then(Item::as_str)
            .map(str::to_owned)
            .ok_or("workspace version must be explicit")
    };
    let coordinated_version = version(&root)? != version(&next_root)?;
    let mut baseline = before.clone();
    if coordinated_version {
        let member_inputs: Vec<_> = members
            .iter()
            .map(|(name, path)| MemberManifest {
                package_name: name.clone(),
                manifest_path: path.clone(),
                text: before[path].clone(),
            })
            .collect();
        let update = plan_version_update(
            &before["Cargo.toml"],
            &member_inputs,
            before.get("Cargo.lock").ok_or("missing Cargo.lock")?,
            &version(&next_root)?,
        )?;
        baseline.insert("Cargo.toml".into(), update.workspace_manifest);
        baseline.insert("Cargo.lock".into(), update.lockfile);
        for member in update.members {
            baseline.insert(member.manifest_path, member.text);
        }
    }
    let baseline_root = parse(&baseline, "Cargo.toml")?;
    let mut allowed_edges = BTreeMap::<String, BTreeSet<String>>::new();
    let mut tools = Vec::new();
    let mut testkit_bench_added = false;
    let mut private_additions = BTreeSet::new();
    if added_tool {
        let tool = parse(after, DEVTOOLS_MANIFEST)?;
        for dependency in ["chrono", "zip", "flate2"] {
            if let Some(spec) = tool.get("dependencies").and_then(|v| v.get(dependency)) {
                if !reviewed_private_addition(dependency, spec) {
                    return Err(format!(
                        "unreviewed private dependency declaration {dependency}"
                    ));
                }
                private_additions.insert(dependency.to_owned());
                tools.push(format!(
                    "{DEVTOOLS_MANIFEST}:{dependency}{}",
                    if dependency == "zip" {
                        "/deflate-flate2"
                    } else {
                        ""
                    }
                ));
            }
        }
    }
    for path in baseline.keys().filter(|path| path.ends_with("Cargo.toml")) {
        let mut prior = parse(&baseline, path)?;
        let mut next = parse(after, path)?;
        if added_tool && path == "Cargo.toml" {
            remove_private_member(&mut next)?;
        }
        let member = members
            .iter()
            .find(|(_, member_path)| *member_path == path)
            .map(|(name, _)| name);
        let mut allowed = if member.is_some_and(|name| name == DEVTOOLS) {
            let edges = normalize_model_runner_migration(&prior, &mut next)?;
            if edges.contains("anyhow") {
                private_additions.insert("anyhow".to_owned());
            }
            tools.extend(edges.iter().map(|name| format!("{path}:{name}")));
            edges
        } else {
            BTreeSet::new()
        };
        allowed.extend(strip_dev(&mut prior, &baseline_root)?);
        allowed.extend(strip_dev(&mut next, &next_root)?);
        if member.is_some_and(|name| name == "ferrum-bench-core") {
            for dependency in ["semver", "toml_edit", "syn", "quote"] {
                let before_dependency = prior
                    .get("dependencies")
                    .and_then(|item| item.get(dependency));
                if let Some(before_spec) = before_dependency {
                    if dependency == "syn" {
                        let before_spec = before_spec.clone();
                        if let Some(mut after_spec) = next
                            .get("dependencies")
                            .and_then(|item| item.get(dependency))
                            .cloned()
                        {
                            if reviewed_syntax_features(&before_spec) == Some(false)
                                && reviewed_syntax_features(&after_spec) == Some(true)
                            {
                                after_spec
                                    .as_table_like_mut()
                                    .expect("reviewed syntax table")
                                    .insert(
                                        "features",
                                        before_spec
                                            .get("features")
                                            .expect("reviewed features")
                                            .clone(),
                                    );
                                // Only the visitor used by the scope parser may
                                // be added. Version/default flags/source remain exact.
                                if semantic(&after_spec) == semantic(&before_spec) {
                                    next.get_mut("dependencies")
                                        .and_then(Item::as_table_like_mut)
                                        .expect("dependency table")
                                        .insert(dependency, before_spec);
                                    tools.push(format!("{path}:{dependency}/visit-mut"));
                                }
                            }
                        }
                    }
                    continue;
                }
                let after_dependency = next
                    .get("dependencies")
                    .and_then(|item| item.get(dependency));
                let Some(spec) = after_dependency else {
                    continue;
                };
                // These closed declarations name the syntax features actually
                // needed by the release scope parser. Other flags remain unknown.
                let reviewed = if dependency == "syn" {
                    reviewed_syntax_features(spec).is_some()
                } else {
                    spec.as_str().is_some()
                };
                if !reviewed {
                    return Err(format!(
                        "unreviewed release-tool dependency declaration {dependency}"
                    ));
                }
                next.get_mut("dependencies")
                    .and_then(Item::as_table_like_mut)
                    .ok_or("missing dependencies")?
                    .remove(dependency);
                allowed.insert(dependency.into());
                tools.push(format!("{path}:{dependency}"));
            }
        }
        if member.is_some_and(|name| name == TESTKIT)
            && prior
                .get("dependencies")
                .and_then(|v| v.get(BENCH_CORE))
                .is_none()
        {
            if let Some(spec) = next.get("dependencies").and_then(|v| v.get(BENCH_CORE)) {
                if !reviewed_local_bench_edge(spec, &next_root, &next_members) {
                    return Err("testkit numerical sharing requires the unchanged local workspace bench-core edge".into());
                }
                isolated_testkit(&baseline, &baseline_root, &members)?;
                isolated_testkit(after, &next_root, &next_members)?;
                next.get_mut("dependencies")
                    .and_then(Item::as_table_like_mut)
                    .ok_or("missing testkit dependencies")?
                    .remove(BENCH_CORE);
                allowed.insert(BENCH_CORE.into());
                tools.push(format!("{path}:{BENCH_CORE}"));
                testkit_bench_added = true;
            }
        }
        if member.is_some_and(|name| name == DEVTOOLS) {
            for dependency in ["chrono", "zip", "flate2"] {
                if prior
                    .get("dependencies")
                    .and_then(|v| v.get(dependency))
                    .is_some()
                {
                    continue;
                }
                let Some(spec) = next.get("dependencies").and_then(|v| v.get(dependency)) else {
                    continue;
                };
                let reviewed = reviewed_private_addition(dependency, spec);
                if !reviewed {
                    return Err(format!(
                        "unreviewed private dependency declaration {dependency}"
                    ));
                }
                next.get_mut("dependencies")
                    .and_then(Item::as_table_like_mut)
                    .ok_or("missing private dependencies")?
                    .remove(dependency);
                allowed.insert(dependency.into());
                private_additions.insert(dependency.to_owned());
                tools.push(format!(
                    "{path}:{dependency}{}",
                    if dependency == "zip" {
                        "/deflate-flate2"
                    } else {
                        ""
                    }
                ));
            }
        }
        if semantic(prior.as_item()) != semantic(next.as_item()) {
            return Err(format!(
                "{path} changes runtime/build/features or unsupported manifest metadata"
            ));
        }
        // If a dependency serves both normal/build and development consumers,
        // an edge/version change is ambiguous and must remain conservative.
        let runtime = runtime_names(&prior, &baseline_root)?;
        allowed.retain(|name| !runtime.contains(name));
        if let Some(member) = member {
            allowed_edges.insert(member.clone(), allowed);
        }
    }
    let prior_lock = parse(&baseline, "Cargo.lock")?;
    let next_lock = parse(after, "Cargo.lock")?;
    let mut prior_header = prior_lock.clone();
    prior_header.as_table_mut().remove("package");
    let mut next_header = next_lock.clone();
    next_header.as_table_mut().remove("package");
    if semantic(prior_header.as_item()) != semantic(next_header.as_item()) {
        return Err("lock metadata changed".into());
    }
    let prior = locks(&prior_lock)?;
    let next = locks(&next_lock)?;
    let local = |records: &BTreeMap<Package, Locked>| {
        records
            .keys()
            .filter(|package| package.source.is_none())
            .map(|package| package.name.clone())
            .collect::<BTreeSet<_>>()
    };
    let member_names: BTreeSet<_> = members.keys().cloned().collect();
    let next_member_names: BTreeSet<_> = next_members.keys().cloned().collect();
    if local(&prior) != member_names || local(&next) != next_member_names {
        return Err("lock local packages differ from workspace members".into());
    }
    if testkit_bench_added {
        let bench = resolve(BENCH_CORE, &next)?;
        let testkit = resolve(TESTKIT, &next)?;
        if bench.source.is_some()
            || testkit.source.is_some()
            || !next[testkit]
                .dependencies
                .iter()
                .any(|edge| resolve(edge, &next).ok() == Some(bench))
        {
            return Err(
                "testkit numerical sharing is missing its local locked bench-core edge".into(),
            );
        }
        for (package, record) in &next {
            if package.source.is_some()
                && record
                    .dependencies
                    .iter()
                    .any(|edge| resolve(edge, &next).ok() == Some(testkit))
            {
                return Err(
                    "registry dependency cannot be a runtime consumer of local testkit".into(),
                );
            }
        }
    }
    // These new private edges reuse exact existing locked identities. They
    // cannot introduce a second version or redirect a shared registry package.
    for name in &private_additions {
        let old = resolve(name, &prior)?;
        let new = resolve(name, &next)?;
        if old != new
            || !old
                .source
                .as_deref()
                .is_some_and(|source| source.starts_with("registry+"))
            || (name == "zip" && old.version != "7.2.0")
        {
            return Err(format!(
                "private tool dependency {name} must reuse its existing registry identity"
            ));
        }
        let tool = resolve(DEVTOOLS, &next)?;
        if !next[tool]
            .dependencies
            .iter()
            .any(|edge| resolve(edge, &next).ok() == Some(new))
        {
            return Err(format!("missing private tool lock edge to {name}"));
        }
    }
    let mut dev_seeds = BTreeSet::new();
    if let Some(expected_dependencies) = &tool_dependencies {
        let tool = resolve(DEVTOOLS, &next)?;
        if tool.source.is_some() {
            return Err("private devtools lock entry must be a local package".into());
        }
        let document = parse(after, DEVTOOLS_MANIFEST)?;
        let tool_version = document
            .get("package")
            .and_then(|item| item.get("version"))
            .ok_or("private devtools version missing")?;
        let expected_version =
            if tool_version.get("workspace").and_then(Item::as_bool) == Some(true) {
                version(&next_root)?
            } else {
                tool_version
                    .as_str()
                    .ok_or("unsupported private devtools version")?
                    .to_owned()
            };
        if tool.version != expected_version {
            return Err("private devtools manifest and lock version differ".into());
        }
        let actual_dependencies: BTreeSet<_> = next[tool]
            .dependencies
            .iter()
            .map(|dependency| resolve(dependency, &next).map(|package| package.name.clone()))
            .collect::<Result<_, _>>()?;
        if &actual_dependencies != expected_dependencies {
            return Err("private devtools manifest and locked dependency names differ".into());
        }
        for (package, record) in &next {
            for dependency in &record.dependencies {
                if resolve(dependency, &next)? == tool {
                    return Err(format!(
                        "locked package {} depends on private devtools",
                        package.name
                    ));
                }
            }
        }
        if added_tool {
            dev_seeds.insert(tool.clone());
        }
    }
    for (package, old) in &prior {
        let new = next.get(package).ok_or_else(|| {
            format!(
                "existing locked package {} {} changed or disappeared",
                package.name, package.version
            )
        })?;
        if old.body != new.body {
            return Err(format!(
                "locked package {} identity/checksum changed",
                package.name
            ));
        }
        if package.source.is_some() {
            if old.dependencies != new.dependencies {
                // The reviewed private ZIP consumer enables exactly one
                // already-locked optional edge. Product manifests and build
                // flags remain unchanged; no private tool is a reverse runtime
                // dependency. This does change workspace-wide feature unification.
                let added: Vec<_> = new.dependencies.difference(&old.dependencies).collect();
                let private_zip_extension = package.name == "zip"
                    && package.version == "7.2.0"
                    && private_additions.contains("zip")
                    && private_additions.contains("flate2")
                    && old.dependencies.is_subset(&new.dependencies)
                    && added.len() == 1
                    && resolve(added[0], &next).is_ok_and(|dependency| {
                        dependency.name == "flate2"
                            && prior.contains_key(dependency)
                            && dependency
                                .source
                                .as_deref()
                                .is_some_and(|source| source.starts_with("registry+"))
                    });
                if !private_zip_extension {
                    return Err(format!(
                        "existing registry dependency graph changed at {}",
                        package.name
                    ));
                }
            }
        } else {
            let allowed = &allowed_edges[&package.name];
            for dependency in old.dependencies.symmetric_difference(&new.dependencies) {
                let changed = resolve(
                    dependency,
                    if new.dependencies.contains(dependency) {
                        &next
                    } else {
                        &prior
                    },
                )?;
                if !allowed.contains(&changed.name) {
                    return Err(format!(
                        "runtime lock edge changed: {} -> {}",
                        package.name, dependency
                    ));
                }
                if new.dependencies.contains(dependency) {
                    dev_seeds.insert(changed.clone());
                }
            }
        }
    }
    let allowed_new = closure(dev_seeds, &next)?;
    for package in next.keys().filter(|package| !prior.contains_key(*package)) {
        if (package.source.is_none() && !(added_tool && package.name == DEVTOOLS))
            || !allowed_new.contains(package)
        {
            return Err(format!("unexplained new lock package {}", package.name));
        }
    }
    Ok(DependencyRefinement {
        paths: after
            .iter()
            .filter_map(|(path, text)| (before.get(path) != Some(text)).then_some(path.clone()))
            .collect(),
        coordinated_version,
        validation_runtime_dependencies: tools,
        private_tool_members: if added_tool {
            vec![DEVTOOLS.into()]
        } else {
            Vec::new()
        },
    })
}

#[cfg(test)]
#[path = "dependency_change_tests.rs"]
mod tests;
