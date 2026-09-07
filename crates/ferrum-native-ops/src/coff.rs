//! Shared, host-independent inspection of MSVC static operator libraries.
//! Archive indexes and undefined import symbols are never export evidence.

use std::collections::{BTreeMap, BTreeSet};

use object::read::archive::{ArchiveFile, ArchiveKind};
use object::read::coff::{CoffBigFile, CoffFile, CoffHeader, ImageSymbol};
use object::{FileKind, Object, ObjectSection, ObjectSymbol, SectionFlags, SymbolFlags};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorObjectIdentity {
    pub format: NativeOperatorObjectFormat,
    pub class_bits: u8,
    pub endianness: NativeOperatorObjectEndianness,
    pub machine: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorObjectFormat {
    Elf,
    MachO,
    Coff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorObjectEndianness {
    Little,
    Big,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorObjectInspection {
    pub identity: NativeOperatorObjectIdentity,
    pub defined_symbols: Vec<String>,
    pub strong_defined_symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorArchiveMember {
    pub name: String,
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub object: NativeOperatorObjectInspection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorArchiveInspection {
    pub identity: NativeOperatorObjectIdentity,
    pub members: Vec<NativeOperatorArchiveMember>,
    pub defined_symbols: Vec<String>,
    pub strong_defined_symbols: Vec<String>,
    pub symbol_definition_counts: BTreeMap<String, usize>,
    pub indexed_symbol_members: BTreeMap<String, Vec<String>>,
}

impl NativeOperatorArchiveInspection {
    /// Even coalescible COMDATs cannot provide two competing ABI entrypoints.
    pub fn require_unique_exports(&self, exports: &[String]) -> Result<(), String> {
        for symbol in exports {
            if self.symbol_definition_counts.get(symbol).copied() != Some(1) {
                return Err(format!(
                    "MSVC archive must define required export exactly once: {symbol}"
                ));
            }
            let defining_member = self
                .members
                .iter()
                .find(|member| member.object.defined_symbols.contains(symbol))
                .expect("definition count checked");
            if !self
                .indexed_symbol_members
                .get(symbol)
                .is_some_and(|members| members.len() == 1 && members[0] == defining_member.name)
            {
                return Err(format!("MSVC linker index does not resolve required export to its real definition: {symbol}"));
            }
        }
        Ok(())
    }
}

pub fn inspect_msvc_object(
    bytes: &[u8],
    expected_target: &str,
) -> Result<NativeOperatorObjectInspection, String> {
    let host = ferrum_types::NativeOperatorHostAbi::for_target(expected_target)?;
    if host.compiler_flavor != ferrum_types::NativeOperatorCompilerFlavor::Msvc {
        return Err("COFF inspection requires an explicit supported MSVC target".into());
    }
    match FileKind::parse(bytes).map_err(|error| format!("invalid COFF object: {error}"))? {
        FileKind::Coff => {
            // The COFF reader also exposes fields used by images; only plain
            // relocatable objects belong in our static source-built archives.
            if bytes.get(16..18) != Some(&[0, 0][..]) {
                return Err("COFF object has an image optional header".into());
            }
            let file: CoffFile<'_> = CoffFile::parse(bytes).map_err(|error| error.to_string())?;
            inspect_coff(file)
        }
        FileKind::CoffBig => {
            inspect_coff(CoffBigFile::parse(bytes).map_err(|error| error.to_string())?)
        }
        FileKind::CoffImport => {
            Err("short COFF import library member is not a native implementation".into())
        }
        other => Err(format!(
            "MSVC native implementation must be COFF/bigobj, got {other:?}"
        )),
    }
}

fn inspect_coff<'data, C: CoffHeader>(
    file: CoffFile<'data, &'data [u8], C>,
) -> Result<NativeOperatorObjectInspection, String> {
    let header = file.coff_header();
    if header.machine() != object::pe::IMAGE_FILE_MACHINE_AMD64 {
        return Err(format!(
            "COFF machine does not match x86_64-pc-windows-msvc: {:#x}",
            header.machine()
        ));
    }
    if header.number_of_sections() == 0 {
        return Err("COFF implementation has no sections".into());
    }
    if header.characteristics()
        & (object::pe::IMAGE_FILE_EXECUTABLE_IMAGE | object::pe::IMAGE_FILE_DLL)
        != 0
    {
        return Err("COFF native member is marked as an executable image".into());
    }
    let mut comdat_sections = BTreeSet::new();
    for section in file.sections() {
        let name = section.name().map_err(|error| error.to_string())?;
        let data = section.data().map_err(|error| error.to_string())?;
        // Long-format import archives consist of ordinary COFF objects. Their
        // .idata sections are authoritative import structure, even when the
        // Object::imports() convenience API returns an empty list.
        if name == ".idata" || name.starts_with(".idata$") {
            return Err(format!("long COFF import library member contains {name}"));
        }
        if name == ".drectve" {
            validate_runtime_directives(data)?;
        }
        if matches!(section.flags(), SectionFlags::Coff { characteristics }
            if characteristics & object::pe::IMAGE_SCN_LNK_COMDAT != 0)
        {
            comdat_sections.insert(section.index().0);
        }
        section
            .coff_relocations()
            .map_err(|error| error.to_string())?;
    }
    let mut selections = BTreeMap::new();
    let mut primary_symbols = BTreeSet::new();
    for symbol in file.symbols() {
        let raw = symbol.coff_symbol();
        if symbol.index().0 + 1 + usize::from(raw.number_of_aux_symbols())
            > header.number_of_symbols() as usize
        {
            return Err("COFF symbol has a truncated auxiliary record".into());
        }
        primary_symbols.insert(symbol.index().0);
        let name = symbol.name().map_err(|error| error.to_string())?;
        if name.starts_with("__IMPORT_DESCRIPTOR_")
            || name == "__NULL_IMPORT_DESCRIPTOR"
            || name.ends_with("_NULL_THUNK_DATA")
        {
            return Err(format!("long COFF import library marker: {name}"));
        }
        if let Some(index) = symbol.section_index() {
            file.section_by_index(index)
                .map_err(|error| error.to_string())?;
            if let SymbolFlags::CoffSection {
                selection,
                associative_section,
            } = symbol.flags()
            {
                if comdat_sections.contains(&index.0) {
                    if !(1..=7).contains(&selection)
                        || selections.insert(index.0, selection).is_some()
                    {
                        return Err("invalid or duplicate COFF COMDAT selection".into());
                    }
                    if selection == object::pe::IMAGE_COMDAT_SELECT_ASSOCIATIVE {
                        let associated =
                            associative_section.ok_or("associative COMDAT has no parent")?;
                        if associated == index || !comdat_sections.contains(&associated.0) {
                            return Err("associative COMDAT has an invalid parent".into());
                        }
                    }
                }
            }
        }
    }
    if comdat_sections
        .iter()
        .any(|index| !selections.contains_key(index))
    {
        return Err("COFF COMDAT section has no valid selection record".into());
    }
    for section in file.sections() {
        for relocation in section
            .coff_relocations()
            .map_err(|error| error.to_string())?
        {
            if relocation.typ.get(object::LittleEndian) != 0
                && !primary_symbols
                    .contains(&(relocation.symbol_table_index.get(object::LittleEndian) as usize))
            {
                return Err("COFF relocation does not reference a real symbol".into());
            }
        }
    }
    let mut defined = BTreeSet::new();
    let mut strong = BTreeSet::new();
    for symbol in file.symbols() {
        if !symbol.is_global() || !symbol.is_definition() {
            continue;
        }
        let name = symbol.name().map_err(|error| error.to_string())?;
        if name.is_empty() || !defined.insert(name.to_owned()) {
            return Err(format!("empty or repeated COFF definition: {name}"));
        }
        let selection = symbol
            .section_index()
            .and_then(|index| selections.get(&index.0))
            .copied();
        if !symbol.is_weak() && matches!(selection, None | Some(1)) {
            strong.insert(name.to_owned());
        }
    }
    Ok(NativeOperatorObjectInspection {
        identity: NativeOperatorObjectIdentity {
            format: NativeOperatorObjectFormat::Coff,
            class_bits: 64,
            endianness: NativeOperatorObjectEndianness::Little,
            machine: u32::from(header.machine()),
        },
        defined_symbols: defined.into_iter().collect(),
        strong_defined_symbols: strong.into_iter().collect(),
    })
}

fn validate_runtime_directives(bytes: &[u8]) -> Result<(), String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "COFF linker directives are not UTF-8")?;
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut quoted = false;
    for ch in text.chars() {
        if ch == '"' {
            quoted = !quoted;
        } else if (ch.is_ascii_whitespace() || ch == '\0') && !quoted {
            if !token.is_empty() {
                tokens.push(std::mem::take(&mut token));
            }
        } else {
            token.push(ch);
        }
    }
    if quoted {
        return Err("COFF linker directive contains an unterminated quote".into());
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    for token in tokens {
        let token = token.to_ascii_lowercase();
        if let Some(library) = token
            .strip_prefix("/defaultlib:")
            .or_else(|| token.strip_prefix("-defaultlib:"))
        {
            let library = library.rsplit(['/', '\\']).next().unwrap_or(library);
            let library = library.strip_suffix(".lib").unwrap_or(library);
            if matches!(
                library,
                "libcmt"
                    | "libcmtd"
                    | "libcpmt"
                    | "libcpmtd"
                    | "msvcrtd"
                    | "msvcprtd"
                    | "vcruntimed"
                    | "ucrtd"
                    | "libvcruntime"
                    | "libvcruntimed"
                    | "libucrt"
                    | "libucrtd"
            ) {
                return Err(format!(
                    "COFF runtime directive conflicts with release /MD: {token}"
                ));
            }
        }
        if let Some(mismatch) = token
            .strip_prefix("/failifmismatch:")
            .or_else(|| token.strip_prefix("-failifmismatch:"))
        {
            if let Some(runtime) = mismatch.strip_prefix("runtimelibrary=") {
                if runtime != "md_dynamicrelease" {
                    return Err(format!(
                        "COFF RuntimeLibrary conflicts with release /MD: {token}"
                    ));
                }
            }
        }
    }
    Ok(())
}

pub fn inspect_msvc_archive(
    bytes: &[u8],
    expected_target: &str,
) -> Result<NativeOperatorArchiveInspection, String> {
    let archive =
        ArchiveFile::parse(bytes).map_err(|error| format!("invalid MSVC archive: {error}"))?;
    if archive.is_thin() || !bytes.starts_with(b"!<arch>\n") {
        return Err("MSVC native archive must be self-contained, not thin".into());
    }
    if archive.kind() != ArchiveKind::Coff {
        return Err("MSVC library is missing its two COFF linker members".into());
    }
    let mut members = Vec::new();
    let mut names = BTreeSet::new();
    let mut counts = BTreeMap::new();
    let mut strong = BTreeSet::new();
    let mut offsets = BTreeMap::new();
    for member in archive.members() {
        let member = member.map_err(|error| error.to_string())?;
        let name =
            std::str::from_utf8(member.name()).map_err(|_| "archive member name is not UTF-8")?;
        if name.is_empty() || name.contains('\0') || !names.insert(name.to_owned()) {
            return Err(format!("empty or duplicate MSVC archive member: {name}"));
        }
        let data = member.data(bytes).map_err(|error| error.to_string())?;
        let offset = member
            .file_range()
            .0
            .checked_sub(60)
            .ok_or("invalid COFF member offset")?;
        offsets.insert(offset, name.to_owned());
        let object = inspect_msvc_object(data, expected_target)
            .map_err(|error| format!("{name}: {error}"))?;
        for symbol in &object.defined_symbols {
            *counts.entry(symbol.clone()).or_insert(0usize) += 1;
        }
        for symbol in &object.strong_defined_symbols {
            if !strong.insert(symbol.clone()) {
                return Err(format!(
                    "duplicate strong MSVC archive definition: {symbol}"
                ));
            }
        }
        members.push(NativeOperatorArchiveMember {
            name: name.to_owned(),
            bytes: data.to_vec(),
            sha256: format!("{:x}", Sha256::digest(data)),
            object,
        });
    }
    let identity = members
        .first()
        .ok_or("MSVC native archive has no object members")?
        .object
        .identity
        .clone();
    let mut indexed_symbol_members: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for symbol in archive
        .symbols()
        .map_err(|error| error.to_string())?
        .ok_or("MSVC archive has no linker symbol table")?
    {
        let symbol = symbol.map_err(|error| error.to_string())?;
        let name =
            std::str::from_utf8(symbol.name()).map_err(|_| "COFF linker symbol is not UTF-8")?;
        if name.is_empty() {
            return Err("COFF linker symbol is empty".into());
        }
        let member = offsets
            .get(&symbol.offset().0)
            .ok_or("COFF linker symbol points outside object members")?;
        indexed_symbol_members
            .entry(name.to_owned())
            .or_default()
            .push(member.clone());
    }
    Ok(NativeOperatorArchiveInspection {
        identity,
        members,
        defined_symbols: counts.keys().cloned().collect(),
        strong_defined_symbols: strong.into_iter().collect(),
        symbol_definition_counts: counts,
        indexed_symbol_members,
    })
}

#[cfg(test)]
#[path = "coff_tests.rs"]
pub(crate) mod tests;
