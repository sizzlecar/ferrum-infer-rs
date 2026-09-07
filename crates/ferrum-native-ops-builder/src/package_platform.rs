//! Platform projection and byte-level verification for native packages.

use super::*;
use source_build::platform;

pub(super) fn source_host_toolchain(
    receipt: &NativeOperatorSourceBuildReceipt,
) -> Result<&NativeOperatorHostToolchainIdentity> {
    let host = &receipt
        .toolchain
        .as_ref()
        .ok_or_else(|| {
            NativeOperatorBuilderError::Invalid("source receipt has no host toolchain".into())
        })?
        .static_identity
        .host_toolchain;
    platform::validate_host_contract(
        host.host_abi.as_ref(),
        &host.target,
        &host.compiler.path,
        &host.environment,
    )?;
    Ok(host)
}

pub(super) fn validate_package_host_link(
    package: &NativeOperatorPackageToolchain,
    source: &NativeOperatorSourceBuildReceipt,
) -> Result<()> {
    let host = source_host_toolchain(source)?;
    platform::validate_host_contract(
        package.host_abi.as_ref(),
        &package.descriptor_target,
        &package.descriptor_compiler.path,
        &package.environment,
    )?;
    if package.host_abi != host.host_abi || package.environment != host.environment {
        return Err(NativeOperatorBuilderError::Invalid(
            "package host ABI/environment differs from the source toolchain".into(),
        ));
    }
    if platform::is_msvc(host.host_abi.as_ref()) {
        let source_archiver = &source
            .toolchain
            .as_ref()
            .expect("validated toolchain")
            .static_identity
            .archiver;
        if !platform::same_path(&package.descriptor_compiler.path, &host.compiler.path)
            || package.descriptor_compiler.sha256 != host.compiler.sha256
            || !platform::same_path(&package.archiver.path, &source_archiver.path)
            || package.archiver.sha256 != source_archiver.sha256
        {
            return Err(NativeOperatorBuilderError::Invalid(
                "MSVC descriptor compiler/archiver differs from the source-build toolchain".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn package_artifact_file(slug: &str, suffix: &str, msvc: bool) -> String {
    platform::archive_file(&format!("libferrum_native_{slug}_{suffix}.a"), msvc)
}

pub(super) fn descriptor_object_file(msvc: bool) -> &'static str {
    if msvc {
        "descriptor.obj"
    } else {
        "descriptor.o"
    }
}

pub(super) fn descriptor_compile_args(msvc: bool) -> Vec<String> {
    if msvc {
        [
            "/nologo",
            "/std:c11",
            "/O2",
            "/MD",
            "/Brepro",
            "/c",
            "descriptor.c",
            "/Fodescriptor.obj",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    } else {
        [
            "-std=c11",
            "-O2",
            "-fno-ident",
            "-fvisibility=hidden",
            "-c",
            "descriptor.c",
            "-o",
            "descriptor.o",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }
}

pub(super) fn descriptor_archive_args(
    artifact: &str,
    source_members: &[NativeOperatorArchiveMemberEvidence],
    msvc: bool,
) -> Vec<String> {
    let mut objects = if msvc {
        source_members
            .iter()
            .map(|member| member.member.clone())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    objects.push(descriptor_object_file(msvc).to_string());
    objects.sort();
    platform::archive_argv("", artifact, objects, msvc)
        .into_iter()
        .skip(1)
        .collect()
}

pub(super) fn render_descriptor_source_for_host(
    export: &str,
    operator: &str,
    operator_abi: &str,
    catalog: &str,
    abi: &str,
    msvc: bool,
) -> String {
    let source = render_descriptor_source(export, operator, operator_abi, catalog, abi);
    if msvc {
        // A public C definition is sufficient for a static COFF library. Adding
        // dllexport would unnecessarily export the internal ABI from the EXE.
        source.replace("__attribute__((visibility(\"default\")))\n", "")
    } else {
        source
    }
}

pub(super) fn package_system_libraries(
    declared: &[NativeOperatorSystemLibrary],
    host_abi: Option<&NativeOperatorHostAbi>,
) -> Result<Vec<NativeOperatorSystemLibrary>> {
    if let Some(abi) = host_abi {
        abi.validate()
            .map_err(NativeOperatorBuilderError::Invalid)?;
    }
    let msvc = platform::is_msvc(host_abi);
    if !msvc && declared.contains(&NativeOperatorSystemLibrary::MsvcRuntime) {
        return Err(NativeOperatorBuilderError::Invalid(
            "MSVC runtime requires an explicit MSVC host ABI".into(),
        ));
    }
    let mut libraries = declared
        .iter()
        .map(|library| {
            if msvc && *library == NativeOperatorSystemLibrary::StdCxx {
                NativeOperatorSystemLibrary::MsvcRuntime
            } else {
                *library
            }
        })
        .collect::<Vec<_>>();
    if msvc {
        libraries.push(NativeOperatorSystemLibrary::MsvcRuntime);
    }
    libraries.sort();
    libraries.dedup();
    Ok(libraries)
}

pub(super) fn recorded_absolute_path(path: &str) -> bool {
    Path::new(path).is_absolute() || platform::normalize_windows_path(path).is_ok()
}

pub(super) fn validate_package_member_host(
    members: &[NativeOperatorArchiveMemberEvidence],
    host_abi: Option<&NativeOperatorHostAbi>,
) -> Result<()> {
    let msvc = platform::is_msvc(host_abi);
    for member in members {
        let identity = &member.object_identity;
        let valid = if msvc {
            member.member.ends_with(".obj")
                && identity.format == NativeOperatorObjectFormat::Coff
                && identity.class_bits == 64
                && identity.machine == 0x8664
                && identity.endianness == NativeOperatorObjectEndianness::Little
        } else {
            member.member.ends_with(".o") && identity.format != NativeOperatorObjectFormat::Coff
        };
        if !valid {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "package member does not match its explicit host ABI: {}",
                member.member,
            )));
        }
    }
    Ok(())
}

pub(super) fn portable_file_name(path: &str) -> Result<&str> {
    let name = platform::basename(path);
    if name.is_empty() || matches!(name, "." | "..") || name.contains(':') {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "path has no valid object filename: {path}"
        )));
    }
    Ok(name)
}

fn read_msvc_archive(path: &Path) -> Result<ferrum_native_ops::NativeOperatorArchiveInspection> {
    let bytes = fs::read(path).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    ferrum_native_ops::inspect_msvc_archive(&bytes, platform::MSVC_TARGET)
        .map_err(NativeOperatorBuilderError::Invalid)
}

pub(super) fn inspect_msvc_package_members(
    archive_path: &Path,
    operator: &str,
    expected: &[NativeOperatorArchiveMemberEvidence],
) -> Result<(Vec<NativeOperatorArchiveMemberEvidence>, String)> {
    validate_archive_member_evidence(operator, expected)?;
    let archive = read_msvc_archive(archive_path)?;
    let mut actual = archive
        .members
        .iter()
        .map(|member| NativeOperatorArchiveMemberEvidence {
            member: member.name.clone(),
            sha256: member.sha256.clone(),
            size_bytes: member.bytes.len() as u64,
            object_identity: member.object.identity.clone(),
        })
        .collect::<Vec<_>>();
    actual.sort_by(|left, right| left.member.cmp(&right.member));
    validate_archive_member_evidence(operator, &actual)?;
    if actual
        .iter()
        .map(|member| &member.member)
        .ne(expected.iter().map(|member| &member.member))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} native archive members differ from expected evidence"
        )));
    }
    if actual != expected {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} native archive member evidence mismatch"
        )));
    }
    let mut log = format!(
        "operator={operator}\narchive_sha256={}\ninspection=MSVC COFF members and linker index\n",
        sha256_file(archive_path)?
    );
    for member in &actual {
        log.push_str(&format!(
            "member={} sha256={} size_bytes={} object_identity={:?}\n",
            member.member, member.sha256, member.size_bytes, member.object_identity
        ));
    }
    Ok((actual, log))
}

pub(super) fn restore_msvc_source_members(
    archive_path: &Path,
    expected: &[NativeOperatorArchiveMemberEvidence],
    staging: &Path,
) -> Result<Vec<PathBuf>> {
    inspect_msvc_package_members(archive_path, "descriptor source", expected)?;
    let archive = read_msvc_archive(archive_path)?;
    let mut paths = Vec::with_capacity(archive.members.len());
    for member in archive.members {
        // Validation above permits only a unique basename. Never strip paths
        // from an untrusted archive member or overwrite an existing object.
        let path = staging.join(&member.name);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: path.clone(),
                source,
            })?;
        file.write_all(&member.bytes)
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: path.clone(),
                source,
            })?;
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests;
