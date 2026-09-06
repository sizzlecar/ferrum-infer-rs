//! Parse raw ldd/otool output. This is a build-host linkage audit; executing the
//! binary and resolving deferred driver dependencies belong to runtime checks.
use super::Backend;
use std::collections::BTreeSet;

/// The supported staging products bind platform and audit parser together.
/// This does not certify an architecture's runtime support or inspect ELF/Mach-O
/// machine headers; those remain properties of the supplied build and binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Format {
    Ldd,
    Otool,
}

impl Format {
    pub(super) fn for_target(backend: Backend, target: &str) -> Result<Self, String> {
        let parts: Vec<_> = target.split('-').collect();
        if parts.iter().any(|part| {
            part.is_empty()
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        }) {
            return Err("target_triple contains an empty or invalid component".into());
        }
        match (backend, parts.as_slice()) {
            (Backend::Cpu | Backend::Cuda, [_, _, "linux", _]) => Ok(Self::Ldd),
            (Backend::Metal, [_, "apple", "darwin"]) => Ok(Self::Otool),
            (Backend::Cpu | Backend::Cuda, _) => {
                Err("CPU and CUDA staging assets require a Linux target and ldd audit".into())
            }
            (Backend::Metal, _) => {
                Err("Metal staging assets require an Apple Darwin target and otool audit".into())
            }
        }
    }
}

pub(super) struct Audit {
    pub runtime_libraries: Vec<String>,
    pub unresolved_runtime_libraries: Vec<String>,
    pub deferred_runtime_dependencies: Vec<String>,
}

fn forbidden(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    ["python", "torch", "vllm"]
        .iter()
        .any(|word| name.contains(word))
}

fn address(value: &str) -> bool {
    value
        .strip_prefix("0x")
        .is_some_and(|hex| !hex.is_empty() && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn library_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._+-/@".contains(&byte))
}

fn ldd_entry(line: &str) -> Result<(String, Option<String>), String> {
    if let Some((name, resolution)) = line.split_once("=>") {
        let name = name.trim();
        let resolution = resolution.trim();
        if !library_token(name) {
            return Err("invalid library name in ldd output".into());
        }
        if resolution == "not found" {
            return Ok((name.into(), None));
        }
        let (path, location) = resolution
            .strip_suffix(')')
            .and_then(|value| value.rsplit_once(" ("))
            .ok_or("ldd resolution must contain a path and load address")?;
        if !path.starts_with('/') || !library_token(path) || !address(location) {
            return Err("invalid resolved library path or address in ldd output".into());
        }
        Ok((name.into(), Some(path.into())))
    } else {
        let (name, location) = line
            .strip_suffix(')')
            .and_then(|value| value.rsplit_once(" ("))
            .ok_or("unrecognized ldd output line")?;
        if !library_token(name) || !address(location) {
            return Err("invalid direct library entry in ldd output".into());
        }
        Ok((name.into(), Some(name.into())))
    }
}

fn otool_entry(line: &str) -> Result<(String, Option<String>), String> {
    let (path, versions) = line
        .strip_suffix(')')
        .and_then(|value| value.split_once(" (compatibility version "))
        .ok_or("unrecognized otool dependency line")?;
    let (compatibility, current) = versions
        .split_once(", current version ")
        .ok_or("otool dependency is missing version information")?;
    let valid_version = |value: &str| {
        !value.is_empty()
            && value
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    };
    if !library_token(path)
        || !(path.starts_with('/') || path.starts_with('@'))
        || !valid_version(compatibility)
        || !valid_version(current)
    {
        return Err("invalid otool library path or version".into());
    }
    Ok((path.into(), Some(path.into())))
}

pub(super) fn inspect(text: &str, backend: Backend, format: Format) -> Result<Audit, String> {
    if text.is_empty()
        || text
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        return Err("dependency audit is empty or contains invalid control characters".into());
    }
    let darwin = format == Format::Otool;
    let mut libraries = BTreeSet::new();
    let mut unresolved = BTreeSet::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        // The workflows prepend `file` output. It is not a dependency, and names
        // in its description (or in a static operator's build metadata) cannot
        // establish dynamic runtime linkage.
        if let Some((_, description)) = line.split_once(": ") {
            if description.starts_with("ELF ") || description.starts_with("Mach-O ") {
                if darwin != description.starts_with("Mach-O ") {
                    return Err("binary file format disagrees with the declared target".into());
                }
                continue;
            }
        }
        // otool also prints the inspected executable path followed by a colon.
        if darwin && line.ends_with(':') && !line.contains(char::is_whitespace) {
            continue;
        }
        let (name, resolved) = if darwin {
            otool_entry(line)?
        } else {
            ldd_entry(line)?
        };
        if forbidden(&name) || resolved.as_deref().is_some_and(forbidden) {
            return Err(format!(
                "forbidden Python/Torch/vLLM dynamic runtime dependency: {name}"
            ));
        }
        if resolved.is_none() {
            if backend != Backend::Cuda || name != "libcuda.so.1" {
                return Err(format!(
                    "unresolved runtime dependency on build host: {name}"
                ));
            }
            unresolved.insert(name.clone());
        }
        libraries.insert(name);
    }
    if libraries.is_empty() {
        return Err("dependency audit contains no dynamic library entries".into());
    }
    let unresolved: Vec<_> = unresolved.into_iter().collect();
    Ok(Audit {
        runtime_libraries: libraries.into_iter().collect(),
        deferred_runtime_dependencies: unresolved.clone(),
        unresolved_runtime_libraries: unresolved,
    })
}
