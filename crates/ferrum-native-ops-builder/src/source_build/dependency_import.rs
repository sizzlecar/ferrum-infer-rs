//! Import a compiler-observed source closure before the ordinary locked build.
//! This is not build evidence: the later source-build still revalidates the
//! exact dependency set and every source/toolchain hash against fresh depfiles.

use super::*;

/// Complete a seed source definition from one NVCC depfile per translation
/// unit, in definition order. Only files inside declared include directories
/// or the unit's own directory may augment its source closure. Absolute paths
/// outside `source_root` are left for the normal build's toolkit/host manifests;
/// this import does not authorize them or authenticate upstream provenance.
pub fn lock_native_operator_source_dependencies(
    definition_path: &Path,
    source_root: &Path,
    depfiles: &[PathBuf],
    output_definition: &Path,
    output_plan: &Path,
) -> Result<NativeOperatorSourceBuildPlan> {
    for output in [output_definition, output_plan] {
        if output.exists() {
            return Err(NativeOperatorBuilderError::OutputExists(
                output.to_path_buf(),
            ));
        }
    }
    if output_definition == output_plan {
        return Err(NativeOperatorBuilderError::Invalid(
            "definition and plan outputs must differ".to_owned(),
        ));
    }
    let mut definition: NativeOperatorSourceDefinition = read_json(definition_path)?;
    validate_definition(&definition)?;
    if depfiles.len() != definition.translation_units.len() {
        return Err(NativeOperatorBuilderError::Invalid(
            "one ordered depfile is required per translation unit".to_owned(),
        ));
    }
    let root = source_root
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: source_root.to_path_buf(),
            source,
        })?;
    let includes = definition
        .include_dirs
        .iter()
        .map(|path| {
            let candidate = root.join(path);
            let canonical =
                candidate
                    .canonicalize()
                    .map_err(|source| NativeOperatorBuilderError::Io {
                        path: candidate,
                        source,
                    })?;
            if !canonical.starts_with(&root) || !canonical.is_dir() {
                return Err(NativeOperatorBuilderError::Invalid(
                    "dependency include directory escapes source root".to_owned(),
                ));
            }
            Ok(canonical)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut headers = BTreeSet::new();
    let mut closures = Vec::new();
    for (index, (unit, depfile)) in definition
        .translation_units
        .iter()
        .zip(depfiles)
        .enumerate()
    {
        let unit_path = locked_source_file(&root, unit)?;
        let raw =
            read_bounded_regular_file(depfile, MAX_DEPFILE_BYTES, "dependency discovery depfile")?;
        let raw = std::str::from_utf8(&raw).map_err(|_| {
            NativeOperatorBuilderError::Invalid(
                "dependency discovery depfile must be UTF-8".to_owned(),
            )
        })?;
        let (_, dependencies) = parse_make_depfile(raw, depfile)?;
        let mut observed = BTreeSet::new();
        let mut saw_unit = false;
        for dependency in dependencies {
            let spelled = Path::new(&dependency);
            let candidate = if spelled.is_absolute() {
                spelled.to_path_buf()
            } else {
                root.join(spelled)
            };
            let canonical =
                candidate
                    .canonicalize()
                    .map_err(|source| NativeOperatorBuilderError::Io {
                        path: candidate.clone(),
                        source,
                    })?;
            if !canonical.is_file() {
                return Err(NativeOperatorBuilderError::Invalid(
                    "dependency is not a regular file".to_owned(),
                ));
            }
            if canonical == unit_path {
                saw_unit = true;
                continue;
            }
            if !canonical.starts_with(&root) {
                // A spelling within the source tree must never escape through
                // symlinks, even though true system headers remain external.
                if spelling_passes_through_source(&candidate, &root)? {
                    return Err(NativeOperatorBuilderError::Invalid(
                        "source dependency symlink escapes root".to_owned(),
                    ));
                }
                continue;
            }
            if !includes
                .iter()
                .any(|directory| canonical.starts_with(directory))
                && !canonical.starts_with(unit_path.parent().expect("unit has parent"))
            {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "source dependency outside declared include roots: {}",
                    canonical.display()
                )));
            }
            let relative =
                path_with_forward_slashes(canonical.strip_prefix(&root).expect("checked root"))?;
            if definition.translation_units.contains(&relative) {
                return Err(NativeOperatorBuilderError::Invalid(
                    "a dependency includes another declared translation unit".to_owned(),
                ));
            }
            observed.insert(relative);
        }
        if !saw_unit {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "depfile does not include its translation unit {unit}"
            )));
        }
        for required in &definition.dependency_closures[index].headers {
            if !observed.contains(required) {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "depfile omits declared required header {required}"
                )));
            }
        }
        headers.extend(observed.iter().cloned());
        closures.push(NativeOperatorTranslationUnitDependencies {
            translation_unit: unit.clone(),
            headers: observed.into_iter().collect(),
        });
    }
    definition.headers = headers.into_iter().collect();
    definition.dependency_closures = closures;
    validate_definition(&definition)?;
    write_json(output_definition, &definition)?;
    lock_native_operator_source_definition(output_definition, &root, output_plan)
}

fn spelling_passes_through_source(candidate: &Path, root: &Path) -> Result<bool> {
    if candidate.starts_with(root) {
        return Ok(true);
    }
    // Resolve ancestor spellings too: e.g. macOS /var and /private/var, or a
    // caller's alias of source_root. Canonicalizing only the final file loses
    // the fact that an in-tree file/directory symlink escaped into the toolkit.
    for ancestor in candidate.ancestors().skip(1) {
        let canonical =
            ancestor
                .canonicalize()
                .map_err(|source| NativeOperatorBuilderError::Io {
                    path: ancestor.to_path_buf(),
                    source,
                })?;
        if canonical.starts_with(root) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let source = root.join("source with spaces #hash");
        fs::create_dir_all(source.join("kernels")).unwrap();
        fs::write(source.join("kernels/unit.cu"), "#include \"required.h\"\n").unwrap();
        fs::write(source.join("kernels/required.h"), "#pragma once\n").unwrap();
        let definition = root.join("seed.json");
        let seed = NativeOperatorSourceDefinition {
            schema_version: NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION,
            operator: CudaNativeBuildUnit::CutlassFp8ScaledMm
                .artifact_operator()
                .to_owned(),
            source_package_kind: "test-source".to_owned(),
            source_package_revision: "test-revision".to_owned(),
            upstream_sources: vec![NativeOperatorUpstreamSource {
                repository: "https://example.com/source.git".to_owned(),
                revision: "fixture".to_owned(),
                license: "MIT".to_owned(),
            }],
            translation_units: vec!["kernels/unit.cu".to_owned()],
            headers: vec!["kernels/required.h".to_owned()],
            dependency_closures: vec![NativeOperatorTranslationUnitDependencies {
                translation_unit: "kernels/unit.cu".to_owned(),
                headers: vec!["kernels/required.h".to_owned()],
            }],
            include_dirs: vec!["kernels".to_owned()],
            defines: vec![],
            nvcc_policy: NativeOperatorNvccPolicy {
                cpp_standard: NativeOperatorCppStandard::Cpp17,
                optimization: NativeOperatorOptimization::O3,
                use_fast_math: false,
                relaxed_constexpr: true,
                extended_lambda: false,
                host_position_independent_code: true,
                host_default_visibility: false,
            },
            architecture: NativeOperatorCudaArchitecture::Sm120ArchitectureSpecific,
            archive_file: "libfixture.a".to_owned(),
        };
        write_json(&definition, &seed).unwrap();
        let input: NativeOperatorSourceDefinition = read_json(&definition).unwrap();
        let depfile = root.join("scan.d");
        let dependencies = input
            .translation_units
            .iter()
            .chain(input.headers.iter())
            .map(|p| source.join(p).display().to_string())
            .collect::<Vec<_>>();
        // The portable serializer requires an absolute target. Use the real
        // host spelling: its Windows branch accepts drive/UNC paths and the
        // parser selects Windows escaping from that target (not from a RHS).
        let target = root.canonicalize().unwrap().join("object.o");
        let target = target.to_str().unwrap();
        let raw = serialize_portable_depfile(target, &dependencies).unwrap();
        assert_eq!(
            parse_make_depfile(std::str::from_utf8(&raw).unwrap(), &depfile).unwrap(),
            (target.to_owned(), dependencies)
        );
        fs::write(&depfile, raw).unwrap();
        (source, definition, depfile)
    }
    fn append_dependency(depfile: &Path, dependency: &Path) {
        let raw = fs::read_to_string(depfile).unwrap();
        let (target, mut dependencies) = parse_make_depfile(&raw, depfile).unwrap();
        dependencies.push(dependency.display().to_string());
        // A synthetic make rule must escape separators, spaces and # on every
        // host. Raw Path::display is not a valid make-word serialization.
        fs::write(
            depfile,
            serialize_portable_depfile(&target, &dependencies).unwrap(),
        )
        .unwrap();
    }
    #[test]
    fn import_locks_observed_closure_without_claiming_a_build() {
        let root = tempfile::tempdir().unwrap();
        let (source, definition, depfile) = fixture(root.path());
        fs::write(source.join("kernels/discovered.h"), "#pragma once\n").unwrap();
        append_dependency(&depfile, &source.join("kernels/discovered.h"));
        let output = root.path().join("observed.json");
        let plan = root.path().join("lock.json");
        let locked = lock_native_operator_source_dependencies(
            &definition,
            &source,
            &[depfile.clone()],
            &output,
            &plan,
        )
        .unwrap();
        assert_eq!(locked.translation_units.len(), 1);
        assert!(!locked.headers.is_empty());
        assert!(locked
            .headers
            .iter()
            .any(|header| header.path == "kernels/discovered.h"));
        assert!(lock_native_operator_source_dependencies(
            &definition,
            &source,
            &[depfile],
            &output,
            &plan
        )
        .is_err());
    }
    #[test]
    fn wrong_unit_or_omitted_required_header_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (source, definition, depfile) = fixture(root.path());
        fs::write(&depfile, "object.o: \n").unwrap();
        assert!(lock_native_operator_source_dependencies(
            &definition,
            &source,
            &[depfile],
            &root.path().join("out.json"),
            &root.path().join("plan.json")
        )
        .is_err());
        assert!(!root.path().join("out.json").exists());
    }
    #[test]
    fn source_dependency_outside_declared_roots_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (source, definition, depfile) = fixture(root.path());
        fs::write(source.join("unlisted.h"), "#pragma once\n").unwrap();
        append_dependency(&depfile, &source.join("unlisted.h"));
        let error = lock_native_operator_source_dependencies(
            &definition,
            &source,
            &[depfile],
            &root.path().join("out.json"),
            &root.path().join("plan.json"),
        )
        .unwrap_err();
        assert!(matches!(error, NativeOperatorBuilderError::Invalid(message)
            if message.contains("outside declared include roots")));
        assert!(!root.path().join("out.json").exists());
    }
    #[cfg(unix)]
    #[test]
    fn source_symlink_cannot_masquerade_as_an_external_toolkit_header() {
        let root = tempfile::tempdir().unwrap();
        let (source, definition, depfile) = fixture(root.path());
        let external = root.path().join("outside.h");
        fs::write(&external, "#pragma once\n").unwrap();
        std::os::unix::fs::symlink(&external, source.join("kernels/link.h")).unwrap();
        let alias = root.path().join("source-alias");
        std::os::unix::fs::symlink(&source, &alias).unwrap();
        append_dependency(&depfile, &alias.join("kernels/link.h"));
        assert!(lock_native_operator_source_dependencies(
            &definition,
            &source,
            &[depfile],
            &root.path().join("out.json"),
            &root.path().join("plan.json")
        )
        .is_err());
    }
}
