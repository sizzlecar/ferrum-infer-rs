use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use syn::{Attribute, ImplItem, Item, ItemMod, TraitItem, UseTree};

const GROUPS: [&str; 4] = ["resource", "execution", "event", "operation"];

#[derive(Debug, Serialize)]
struct DependencyGraphArtifact {
    schema_version: u32,
    artifact_type: &'static str,
    parser: ParserPolicy,
    groups: Vec<GroupGraph>,
    summary: ArtifactSummary,
}

#[derive(Debug, Serialize)]
struct ParserPolicy {
    kind: &'static str,
    dependency_scope: &'static str,
    cfg_test_subtrees_excluded: bool,
    internal_glob_policy: &'static str,
    unresolved_internal_reference_policy: &'static str,
    hidden_production_module_policy: &'static str,
    facade_semantic_item_policy: &'static str,
}

#[derive(Debug, Serialize)]
struct GroupGraph {
    group: String,
    facade: String,
    owners: Vec<String>,
    edges: Vec<DependencyEdge>,
    strongly_connected_components: Vec<Vec<String>>,
    multi_module_sccs: Vec<Vec<String>>,
    topological_order: Vec<String>,
    diagnostics: GraphDiagnostics,
    summary: GroupSummary,
}

#[derive(Debug, Serialize)]
struct DependencyEdge {
    importer: String,
    dependency: String,
    evidence: Vec<DependencyEvidence>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct DependencyEvidence {
    path: String,
    kind: String,
    reference: String,
}

#[derive(Clone, Debug, Default, Serialize)]
struct GraphDiagnostics {
    unresolved_internal_references: Vec<String>,
    ambiguous_internal_references: Vec<String>,
    unsupported_internal_globs: Vec<String>,
    facade_owned_items: Vec<String>,
    facade_owned_references: Vec<String>,
    hidden_production_modules: Vec<String>,
}

impl GraphDiagnostics {
    fn normalize(&mut self) {
        self.unresolved_internal_references.sort();
        self.unresolved_internal_references.dedup();
        self.ambiguous_internal_references.sort();
        self.ambiguous_internal_references.dedup();
        self.unsupported_internal_globs.sort();
        self.unsupported_internal_globs.dedup();
        self.facade_owned_items.sort();
        self.facade_owned_items.dedup();
        self.facade_owned_references.sort();
        self.facade_owned_references.dedup();
        self.hidden_production_modules.sort();
        self.hidden_production_modules.dedup();
    }

    fn count(&self) -> usize {
        self.unresolved_internal_references.len()
            + self.ambiguous_internal_references.len()
            + self.unsupported_internal_globs.len()
            + self.facade_owned_items.len()
            + self.facade_owned_references.len()
            + self.hidden_production_modules.len()
    }
}

#[derive(Debug, Serialize)]
struct GroupSummary {
    owner_count: usize,
    edge_count: usize,
    multi_module_scc_count: usize,
    diagnostic_count: usize,
    pass: bool,
}

#[derive(Debug, Serialize)]
struct ArtifactSummary {
    production_group_count: usize,
    owner_count: usize,
    edge_count: usize,
    multi_module_scc_count: usize,
    diagnostic_count: usize,
    pass: bool,
}

struct OwnerSource {
    name: String,
    relative_path: String,
    syntax: syn::File,
}

struct GroupContext<'a> {
    group: &'a str,
    importer: &'a str,
    source_path: &'a str,
    module_names: &'a BTreeSet<String>,
    symbol_owners: &'a BTreeMap<String, BTreeSet<String>>,
    facade_bindings: &'a BTreeSet<String>,
    facade_owned: &'a BTreeSet<String>,
    edges: &'a mut BTreeMap<(String, String), BTreeSet<DependencyEvidence>>,
    diagnostics: &'a mut GraphDiagnostics,
}

impl GroupContext<'_> {
    fn record_edge(&mut self, dependency: &str, kind: &str, reference: &str) {
        if dependency == self.importer {
            return;
        }
        self.edges
            .entry((self.importer.to_string(), dependency.to_string()))
            .or_default()
            .insert(DependencyEvidence {
                path: self.source_path.to_string(),
                kind: kind.to_string(),
                reference: reference.to_string(),
            });
    }

    fn resolve_symbol(&mut self, symbol: &str, kind: &str, reference: &str, strict: bool) {
        if self.facade_owned.contains(symbol) {
            self.diagnostics.facade_owned_references.push(format!(
                "{}: {} references facade-owned `{symbol}` via `{reference}`",
                self.source_path, self.importer
            ));
            return;
        }
        match self.symbol_owners.get(symbol) {
            Some(owners) if owners.len() == 1 => {
                self.record_edge(owners.first().expect("one owner"), kind, reference);
            }
            Some(owners) => self.diagnostics.ambiguous_internal_references.push(format!(
                "{}: `{reference}` resolves `{symbol}` to {:?}",
                self.source_path, owners
            )),
            None if strict && !self.facade_bindings.contains(symbol) => self
                .diagnostics
                .unresolved_internal_references
                .push(format!(
                    "{}: `{reference}` does not resolve in facade `{}`",
                    self.source_path, self.group
                )),
            None => {}
        }
    }

    fn resolve_path(&mut self, segments: &[String], kind: &str, reference: &str) {
        if segments.is_empty() {
            return;
        }
        if segments[0] == "super" {
            let parent_depth = segments
                .iter()
                .take_while(|segment| *segment == "super")
                .count();
            let rest = &segments[parent_depth..];
            if parent_depth == 1 {
                let Some(first) = rest.first() else {
                    return;
                };
                if self.module_names.contains(first) {
                    self.record_edge(first, kind, reference);
                } else {
                    self.resolve_symbol(first, kind, reference, true);
                }
            } else if let Some(symbol) = rest.last() {
                self.resolve_symbol(symbol, kind, reference, false);
            }
            return;
        }
        if segments.first().is_some_and(|segment| segment == "crate")
            && segments.get(1).is_some_and(|segment| segment == "vnext")
        {
            let rest = &segments[2..];
            if rest.first().is_some_and(|segment| segment == self.group) {
                if let Some(owner) = rest
                    .get(1)
                    .filter(|owner| self.module_names.contains(*owner))
                {
                    self.record_edge(owner, kind, reference);
                    return;
                }
            }
            if let Some(symbol) = rest.last() {
                self.resolve_symbol(symbol, kind, reference, false);
            }
        }
    }
}

struct ReferenceVisitor<'a> {
    context: GroupContext<'a>,
}

impl<'ast> Visit<'ast> for ReferenceVisitor<'_> {
    fn visit_item(&mut self, item: &'ast Item) {
        if item_is_cfg_test(item) {
            return;
        }
        visit::visit_item(self, item);
    }

    fn visit_impl_item(&mut self, item: &'ast ImplItem) {
        if impl_item_is_cfg_test(item) {
            return;
        }
        visit::visit_impl_item(self, item);
    }

    fn visit_trait_item(&mut self, item: &'ast TraitItem) {
        if trait_item_is_cfg_test(item) {
            return;
        }
        visit::visit_trait_item(self, item);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        let mut leaves = Vec::new();
        flatten_use_tree(&item.tree, &mut Vec::new(), &mut leaves);
        for leaf in leaves {
            let reference = leaf.segments.join("::");
            if leaf.glob {
                if path_targets_group(&leaf.segments, self.context.group) {
                    self.context
                        .diagnostics
                        .unsupported_internal_globs
                        .push(format!(
                            "{}: internal glob `{reference}::*` is unsupported",
                            self.context.source_path
                        ));
                }
                continue;
            }
            self.context.resolve_path(&leaf.segments, "use", &reference);
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        let segments = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>();
        let reference = segments.join("::");
        self.context.resolve_path(&segments, "path", &reference);
        visit::visit_path(self, path);
    }
}

#[derive(Debug)]
struct UseLeaf {
    segments: Vec<String>,
    glob: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vNext owner dependency graph failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err(format!(
            "usage: vnext_owner_dependency_graph <current-vnext-dir> <output-json>; got {} arguments",
            args.len()
        ));
    }
    let current_root = PathBuf::from(&args[0]);
    let output_path = PathBuf::from(&args[1]);
    let mut groups = Vec::new();
    for group in GROUPS {
        groups.push(load_group(&current_root, group)?);
    }
    let owner_count = groups.iter().map(|group| group.summary.owner_count).sum();
    let edge_count = groups.iter().map(|group| group.summary.edge_count).sum();
    let multi_module_scc_count = groups
        .iter()
        .map(|group| group.summary.multi_module_scc_count)
        .sum();
    let diagnostic_count = groups
        .iter()
        .map(|group| group.summary.diagnostic_count)
        .sum();
    let pass = groups.len() == GROUPS.len()
        && owner_count > 0
        && multi_module_scc_count == 0
        && diagnostic_count == 0
        && groups.iter().all(|group| group.summary.pass);
    let artifact = DependencyGraphArtifact {
        schema_version: 1,
        artifact_type: "runtime_vnext_s0a_owner_dependency_graph",
        parser: ParserPolicy {
            kind: "syn_ast",
            dependency_scope: "complete_intra_facade_owner_graph",
            cfg_test_subtrees_excluded: true,
            internal_glob_policy: "reject",
            unresolved_internal_reference_policy: "reject",
            hidden_production_module_policy: "reject",
            facade_semantic_item_policy: "reject",
        },
        groups,
        summary: ArtifactSummary {
            production_group_count: GROUPS.len(),
            owner_count,
            edge_count,
            multi_module_scc_count,
            diagnostic_count,
            pass,
        },
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(
        &output_path,
        serde_json::to_vec_pretty(&artifact)
            .map_err(|error| format!("serialize owner dependency graph: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    println!(
        "VNEXT OWNER DEPENDENCY GRAPH {}: groups={} owners={} edges={} scc={} diagnostics={} output={}",
        if pass { "PASS" } else { "FAIL" },
        artifact.summary.production_group_count,
        owner_count,
        edge_count,
        multi_module_scc_count,
        diagnostic_count,
        output_path.display()
    );
    if pass {
        Ok(())
    } else {
        Err("owner dependency graph acceptance criteria failed; inspect emitted JSON".to_string())
    }
}

fn load_group(current_root: &Path, group: &str) -> Result<GroupGraph, String> {
    let facade_path = current_root.join(format!("{group}.rs"));
    let facade = parse_file(&facade_path)?;
    let owners = load_owner_sources(current_root, group, &facade)?;
    analyze_group(group, &format!("{group}.rs"), &facade, owners)
}

fn load_owner_sources(
    current_root: &Path,
    group: &str,
    facade: &syn::File,
) -> Result<Vec<OwnerSource>, String> {
    let mut owners = Vec::new();
    for item in &facade.items {
        let Item::Mod(module) = item else {
            continue;
        };
        if cfg_test_only(&module.attrs) {
            continue;
        }
        if module.content.is_some() {
            return Err(format!(
                "{group}.rs: production owner `{}` must be a physical peer module",
                module.ident
            ));
        }
        let name = module.ident.to_string();
        let path = resolve_owner_path(current_root, group, module)?;
        owners.push(OwnerSource {
            name,
            relative_path: relative_path(current_root, &path)?,
            syntax: parse_file(&path)?,
        });
    }
    owners.sort_by(|left, right| left.name.cmp(&right.name));
    if owners.is_empty() {
        return Err(format!("{group}.rs declares no production owner modules"));
    }
    let unique = owners
        .iter()
        .map(|owner| owner.name.as_str())
        .collect::<BTreeSet<_>>();
    if unique.len() != owners.len() {
        return Err(format!(
            "{group}.rs declares duplicate production owner modules"
        ));
    }
    Ok(owners)
}

fn resolve_owner_path(
    current_root: &Path,
    group: &str,
    module: &ItemMod,
) -> Result<PathBuf, String> {
    if let Some(path) = explicit_module_path(&module.attrs)? {
        let candidate = current_root.join(path);
        if candidate.is_file() {
            return Ok(candidate);
        }
        return Err(format!(
            "{group}.rs: module `{}` path does not exist: {}",
            module.ident,
            candidate.display()
        ));
    }
    let flat = current_root
        .join(group)
        .join(format!("{}.rs", module.ident));
    let nested = current_root
        .join(group)
        .join(module.ident.to_string())
        .join("mod.rs");
    match (flat.is_file(), nested.is_file()) {
        (true, false) => Ok(flat),
        (false, true) => Ok(nested),
        (false, false) => Err(format!(
            "{group}.rs: no source for production owner `{}`",
            module.ident
        )),
        (true, true) => Err(format!(
            "{group}.rs: ambiguous source for production owner `{}`",
            module.ident
        )),
    }
}

fn analyze_group(
    group: &str,
    facade_path: &str,
    facade: &syn::File,
    owners: Vec<OwnerSource>,
) -> Result<GroupGraph, String> {
    let module_names = owners
        .iter()
        .map(|owner| owner.name.clone())
        .collect::<BTreeSet<_>>();
    let mut diagnostics = GraphDiagnostics::default();
    let facade_owned = collect_facade_owned_items(facade_path, facade, &mut diagnostics);
    let facade_bindings = collect_facade_bindings(facade);
    let mut symbol_owners = BTreeMap::<String, BTreeSet<String>>::new();
    for owner in &owners {
        collect_owner_symbols(&owner.syntax.items, &owner.name, &mut symbol_owners);
        collect_hidden_modules(
            &owner.syntax.items,
            &owner.relative_path,
            &mut diagnostics.hidden_production_modules,
        );
    }

    let mut edge_map = BTreeMap::<(String, String), BTreeSet<DependencyEvidence>>::new();
    for owner in &owners {
        let mut visitor = ReferenceVisitor {
            context: GroupContext {
                group,
                importer: &owner.name,
                source_path: &owner.relative_path,
                module_names: &module_names,
                symbol_owners: &symbol_owners,
                facade_bindings: &facade_bindings,
                facade_owned: &facade_owned,
                edges: &mut edge_map,
                diagnostics: &mut diagnostics,
            },
        };
        for item in &owner.syntax.items {
            visitor.visit_item(item);
        }
    }
    diagnostics.normalize();
    let edges = edge_map
        .iter()
        .map(|((importer, dependency), evidence)| DependencyEdge {
            importer: importer.clone(),
            dependency: dependency.clone(),
            evidence: evidence.iter().cloned().collect(),
        })
        .collect::<Vec<_>>();
    let owner_names = module_names.iter().cloned().collect::<Vec<_>>();
    let edge_pairs = edge_map.keys().cloned().collect::<BTreeSet<_>>();
    let strongly_connected_components = strongly_connected_components(&owner_names, &edge_pairs);
    let multi_module_sccs = strongly_connected_components
        .iter()
        .filter(|component| component.len() > 1)
        .cloned()
        .collect::<Vec<_>>();
    let topological_order = dependencies_first_topological_order(&owner_names, &edge_pairs);
    let diagnostic_count = diagnostics.count();
    let pass = !owner_names.is_empty()
        && multi_module_sccs.is_empty()
        && diagnostic_count == 0
        && topological_order.len() == owner_names.len();
    Ok(GroupGraph {
        group: group.to_string(),
        facade: facade_path.to_string(),
        owners: owner_names,
        edges,
        strongly_connected_components,
        multi_module_sccs: multi_module_sccs.clone(),
        topological_order,
        summary: GroupSummary {
            owner_count: module_names.len(),
            edge_count: edge_map.len(),
            multi_module_scc_count: multi_module_sccs.len(),
            diagnostic_count,
            pass,
        },
        diagnostics,
    })
}

fn collect_owner_symbols(
    items: &[Item],
    owner: &str,
    symbols: &mut BTreeMap<String, BTreeSet<String>>,
) {
    for item in items {
        if item_is_cfg_test(item) {
            continue;
        }
        let name = match item {
            Item::Const(item) => Some(item.ident.to_string()),
            Item::Enum(item) => Some(item.ident.to_string()),
            Item::Fn(item) => Some(item.sig.ident.to_string()),
            Item::Mod(item) if item.content.is_some() => Some(item.ident.to_string()),
            Item::Static(item) => Some(item.ident.to_string()),
            Item::Struct(item) => Some(item.ident.to_string()),
            Item::Trait(item) => Some(item.ident.to_string()),
            Item::TraitAlias(item) => Some(item.ident.to_string()),
            Item::Type(item) => Some(item.ident.to_string()),
            Item::Union(item) => Some(item.ident.to_string()),
            Item::Macro(item) if item.ident.is_some() => {
                item.ident.as_ref().map(ToString::to_string)
            }
            Item::Macro(item) => first_macro_identifier(&item.mac.tokens),
            _ => None,
        };
        if let Some(name) = name {
            symbols.entry(name).or_default().insert(owner.to_string());
        }
    }
}

fn collect_facade_owned_items(
    facade_path: &str,
    facade: &syn::File,
    diagnostics: &mut GraphDiagnostics,
) -> BTreeSet<String> {
    let mut owned = BTreeSet::new();
    for item in &facade.items {
        if item_is_cfg_test(item)
            || matches!(item, Item::Use(_) | Item::Mod(_) | Item::ExternCrate(_))
        {
            continue;
        }
        let (kind, name) = match item {
            Item::Const(item) => ("const", item.ident.to_string()),
            Item::Enum(item) => ("enum", item.ident.to_string()),
            Item::Fn(item) => ("function", item.sig.ident.to_string()),
            Item::Static(item) => ("static", item.ident.to_string()),
            Item::Struct(item) => ("struct", item.ident.to_string()),
            Item::Trait(item) => ("trait", item.ident.to_string()),
            Item::TraitAlias(item) => ("trait_alias", item.ident.to_string()),
            Item::Type(item) => ("type_alias", item.ident.to_string()),
            Item::Union(item) => ("union", item.ident.to_string()),
            Item::Impl(_) => ("impl", "<impl>".to_string()),
            Item::Macro(item) => (
                "macro",
                item.ident
                    .as_ref()
                    .map(ToString::to_string)
                    .or_else(|| first_macro_identifier(&item.mac.tokens))
                    .unwrap_or_else(|| "<macro>".to_string()),
            ),
            other => (
                "unsupported",
                format!("{:?}", std::mem::discriminant(other)),
            ),
        };
        owned.insert(name.clone());
        diagnostics
            .facade_owned_items
            .push(format!("{facade_path}: {kind} `{name}`"));
    }
    owned
}

fn collect_facade_bindings(facade: &syn::File) -> BTreeSet<String> {
    let mut bindings = BTreeSet::new();
    for item in &facade.items {
        let Item::Use(item) = item else {
            continue;
        };
        let mut leaves = Vec::new();
        flatten_use_tree(&item.tree, &mut Vec::new(), &mut leaves);
        for leaf in leaves.into_iter().filter(|leaf| !leaf.glob) {
            if let Some(binding) = leaf.segments.last() {
                if binding != "self" {
                    bindings.insert(binding.clone());
                }
            }
        }
    }
    bindings
}

fn collect_hidden_modules(items: &[Item], path: &str, diagnostics: &mut Vec<String>) {
    for item in items {
        if item_is_cfg_test(item) {
            continue;
        }
        let Item::Mod(module) = item else {
            continue;
        };
        if let Some((_, nested)) = &module.content {
            collect_hidden_modules(nested, path, diagnostics);
        } else {
            diagnostics.push(format!(
                "{path}: external production submodule `{}` is hidden below its owner",
                module.ident
            ));
        }
    }
}

fn flatten_use_tree(tree: &UseTree, prefix: &mut Vec<String>, leaves: &mut Vec<UseLeaf>) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            flatten_use_tree(&path.tree, prefix, leaves);
            prefix.pop();
        }
        UseTree::Name(name) => {
            let mut segments = prefix.clone();
            segments.push(name.ident.to_string());
            leaves.push(UseLeaf {
                segments,
                glob: false,
            });
        }
        UseTree::Rename(rename) => {
            let mut segments = prefix.clone();
            segments.push(rename.ident.to_string());
            leaves.push(UseLeaf {
                segments,
                glob: false,
            });
        }
        UseTree::Glob(_) => leaves.push(UseLeaf {
            segments: prefix.clone(),
            glob: true,
        }),
        UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(item, prefix, leaves);
            }
        }
    }
}

fn path_targets_group(segments: &[String], group: &str) -> bool {
    if segments.first().is_some_and(|segment| segment == "super") {
        return true;
    }
    segments.first().is_some_and(|segment| segment == "crate")
        && segments.get(1).is_some_and(|segment| segment == "vnext")
        && segments.get(2).is_none_or(|segment| segment == group)
}

fn strongly_connected_components(
    nodes: &[String],
    edges: &BTreeSet<(String, String)>,
) -> Vec<Vec<String>> {
    struct Tarjan<'a> {
        adjacency: &'a BTreeMap<String, Vec<String>>,
        next_index: usize,
        indices: BTreeMap<String, usize>,
        lowlinks: BTreeMap<String, usize>,
        stack: Vec<String>,
        on_stack: BTreeSet<String>,
        components: Vec<Vec<String>>,
    }

    impl Tarjan<'_> {
        fn visit(&mut self, node: &str) {
            let index = self.next_index;
            self.next_index += 1;
            self.indices.insert(node.to_string(), index);
            self.lowlinks.insert(node.to_string(), index);
            self.stack.push(node.to_string());
            self.on_stack.insert(node.to_string());

            for dependency in self.adjacency.get(node).into_iter().flatten() {
                if !self.indices.contains_key(dependency) {
                    self.visit(dependency);
                    let dependency_low = self.lowlinks[dependency];
                    let node_low = self.lowlinks[node];
                    self.lowlinks
                        .insert(node.to_string(), node_low.min(dependency_low));
                } else if self.on_stack.contains(dependency) {
                    let dependency_index = self.indices[dependency];
                    let node_low = self.lowlinks[node];
                    self.lowlinks
                        .insert(node.to_string(), node_low.min(dependency_index));
                }
            }

            if self.lowlinks[node] == self.indices[node] {
                let mut component = Vec::new();
                loop {
                    let member = self.stack.pop().expect("Tarjan stack cannot underflow");
                    self.on_stack.remove(&member);
                    let at_root = member == node;
                    component.push(member);
                    if at_root {
                        break;
                    }
                }
                component.sort();
                self.components.push(component);
            }
        }
    }

    let mut adjacency = nodes
        .iter()
        .map(|node| (node.clone(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for (importer, dependency) in edges {
        adjacency
            .entry(importer.clone())
            .or_default()
            .push(dependency.clone());
    }
    for dependencies in adjacency.values_mut() {
        dependencies.sort();
        dependencies.dedup();
    }
    let mut tarjan = Tarjan {
        adjacency: &adjacency,
        next_index: 0,
        indices: BTreeMap::new(),
        lowlinks: BTreeMap::new(),
        stack: Vec::new(),
        on_stack: BTreeSet::new(),
        components: Vec::new(),
    };
    for node in nodes {
        if !tarjan.indices.contains_key(node) {
            tarjan.visit(node);
        }
    }
    tarjan.components.sort();
    tarjan.components
}

fn dependencies_first_topological_order(
    nodes: &[String],
    edges: &BTreeSet<(String, String)>,
) -> Vec<String> {
    let mut remaining_dependencies = nodes
        .iter()
        .map(|node| (node.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = nodes
        .iter()
        .map(|node| (node.clone(), BTreeSet::<String>::new()))
        .collect::<BTreeMap<_, _>>();
    for (importer, dependency) in edges {
        *remaining_dependencies.entry(importer.clone()).or_default() += 1;
        dependents
            .entry(dependency.clone())
            .or_default()
            .insert(importer.clone());
    }
    let mut ready = remaining_dependencies
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(node, _)| node.clone())
        .collect::<BTreeSet<_>>();
    let mut order = Vec::new();
    while let Some(node) = ready.pop_first() {
        order.push(node.clone());
        for dependent in dependents.get(&node).into_iter().flatten() {
            let count = remaining_dependencies
                .get_mut(dependent)
                .expect("dependent must be a graph node");
            *count -= 1;
            if *count == 0 {
                ready.insert(dependent.clone());
            }
        }
    }
    order
}

fn item_is_cfg_test(item: &Item) -> bool {
    let attrs = match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        _ => return false,
    };
    cfg_test_only(attrs)
}

fn impl_item_is_cfg_test(item: &ImplItem) -> bool {
    let attrs = match item {
        ImplItem::Const(item) => &item.attrs,
        ImplItem::Fn(item) => &item.attrs,
        ImplItem::Macro(item) => &item.attrs,
        ImplItem::Type(item) => &item.attrs,
        _ => return false,
    };
    cfg_test_only(attrs)
}

fn trait_item_is_cfg_test(item: &TraitItem) -> bool {
    let attrs = match item {
        TraitItem::Const(item) => &item.attrs,
        TraitItem::Fn(item) => &item.attrs,
        TraitItem::Macro(item) => &item.attrs,
        TraitItem::Type(item) => &item.attrs,
        _ => return false,
    };
    cfg_test_only(attrs)
}

fn cfg_test_only(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        attr.meta
            .require_list()
            .is_ok_and(|list| list.tokens.to_string() == "test")
    })
}

fn explicit_module_path(attrs: &[Attribute]) -> Result<Option<PathBuf>, String> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        let value = attr
            .meta
            .require_name_value()
            .map_err(|error| format!("invalid #[path] attribute: {error}"))?;
        let syn::Expr::Lit(expression) = &value.value else {
            return Err("#[path] value must be a string literal".to_string());
        };
        let syn::Lit::Str(path) = &expression.lit else {
            return Err("#[path] value must be a string literal".to_string());
        };
        return Ok(Some(PathBuf::from(path.value())));
    }
    Ok(None)
}

fn first_macro_identifier(tokens: &proc_macro2::TokenStream) -> Option<String> {
    tokens.clone().into_iter().find_map(|token| match token {
        proc_macro2::TokenTree::Ident(ident) => Some(ident.to_string()),
        _ => None,
    })
}

fn parse_file(path: &Path) -> Result<syn::File, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    syn::parse_file(&source).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn relative_path(root: &Path, path: &Path) -> Result<String, String> {
    path.strip_prefix(root)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .map_err(|error| {
            format!(
                "{} is not below {}: {error}",
                path.display(),
                root.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(name: &str, source: &str) -> OwnerSource {
        OwnerSource {
            name: name.to_string(),
            relative_path: format!("sample/{name}.rs"),
            syntax: syn::parse_file(source).unwrap(),
        }
    }

    fn graph(facade: &str, owners: Vec<OwnerSource>) -> GroupGraph {
        analyze_group(
            "sample",
            "sample.rs",
            &syn::parse_file(facade).unwrap(),
            owners,
        )
        .unwrap()
    }

    #[test]
    fn grouped_and_renamed_imports_produce_dependencies_first_order() {
        let graph = graph(
            "mod a; mod b; mod c; pub use a::*; pub use b::*; pub use c::*;",
            vec![
                owner("a", "pub struct A;"),
                owner("b", "use super::{A as Root}; pub struct B(Root);"),
                owner("c", "use super::{B}; pub struct C(B);"),
            ],
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .map(|edge| (edge.importer.as_str(), edge.dependency.as_str()))
                .collect::<Vec<_>>(),
            vec![("b", "a"), ("c", "b")]
        );
        assert_eq!(graph.topological_order, ["a", "b", "c"]);
        assert!(graph.summary.pass);
    }

    #[test]
    fn tarjan_reports_a_multi_owner_cycle() {
        let graph = graph(
            "mod a; mod b; pub use a::*; pub use b::*;",
            vec![
                owner("a", "use super::B; pub struct A(pub Option<B>);"),
                owner("b", "use super::A; pub struct B(pub Option<A>);"),
            ],
        );
        assert_eq!(graph.multi_module_sccs, [vec!["a", "b"]]);
        assert!(graph.topological_order.is_empty());
        assert!(!graph.summary.pass);
    }

    #[test]
    fn cfg_test_subtree_does_not_create_an_edge() {
        let graph = graph(
            "mod a; mod b; pub use a::*; pub use b::*;",
            vec![
                owner(
                    "a",
                    "pub struct A; #[cfg(test)] mod tests { use super::super::B; }",
                ),
                owner("b", "pub struct B;"),
            ],
        );
        assert!(graph.edges.is_empty());
        assert!(graph.summary.pass);
    }

    #[test]
    fn hidden_modules_globs_and_facade_semantics_fail_closed() {
        let graph = graph(
            "mod a; mod b; pub use a::*; pub use b::*; pub const LIMIT: usize = 1;",
            vec![
                owner("a", "mod hidden; use super::*; pub struct A;"),
                owner("b", "use super::LIMIT; pub struct B;"),
            ],
        );
        assert_eq!(graph.diagnostics.hidden_production_modules.len(), 1);
        assert_eq!(graph.diagnostics.unsupported_internal_globs.len(), 1);
        assert_eq!(graph.diagnostics.facade_owned_items.len(), 1);
        assert_eq!(graph.diagnostics.facade_owned_references.len(), 1);
        assert!(!graph.summary.pass);
    }

    #[test]
    fn unresolved_and_ambiguous_internal_references_fail_closed() {
        let graph = graph(
            "mod a; mod b; mod c; pub use a::*; pub use b::*; pub use c::*;",
            vec![
                owner("a", "pub struct Shared;"),
                owner("b", "pub struct Shared;"),
                owner("c", "use super::{Missing, Shared}; pub struct C;"),
            ],
        );
        assert_eq!(graph.diagnostics.unresolved_internal_references.len(), 1);
        assert_eq!(graph.diagnostics.ambiguous_internal_references.len(), 1);
        assert!(!graph.summary.pass);
    }
}
