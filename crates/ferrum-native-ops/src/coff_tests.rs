use super::*;

const TARGET: &str = "x86_64-pc-windows-msvc";

#[derive(Clone)]
struct SectionFixture {
    name: &'static str,
    data: Vec<u8>,
    comdat: Option<u8>,
}

#[derive(Clone)]
struct SymbolFixture {
    name: String,
    section: u32,
    storage: u8,
}

struct ObjectFixture {
    big: bool,
    machine: u16,
    sections: Vec<SectionFixture>,
    symbols: Vec<SymbolFixture>,
}

impl ObjectFixture {
    fn code(exports: &[&str]) -> Self {
        Self {
            big: false,
            machine: 0x8664,
            sections: vec![SectionFixture {
                name: ".text",
                data: vec![0x31, 0xc0, 0xc3],
                comdat: None,
            }],
            symbols: exports
                .iter()
                .map(|name| SymbolFixture {
                    name: (*name).to_string(),
                    section: 1,
                    storage: 2,
                })
                .collect(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let header_len = if self.big { 56 } else { 20 };
        let symbol_len = if self.big { 20 } else { 18 };
        let mut bytes = vec![0; header_len + 40 * self.sections.len()];
        if self.big {
            put16(&mut bytes, 2, 0xffff);
            put16(&mut bytes, 4, 2);
            put16(&mut bytes, 6, self.machine);
            bytes[12..28].copy_from_slice(&[
                0xc7, 0xa1, 0xba, 0xd1, 0xee, 0xba, 0xa9, 0x4b, 0xaf, 0x20, 0xfa, 0xf6, 0x6a, 0xa4,
                0xdc, 0xb8,
            ]);
            put32(&mut bytes, 44, self.sections.len() as u32);
        } else {
            put16(&mut bytes, 0, self.machine);
            put16(&mut bytes, 2, self.sections.len() as u16);
        }
        for (i, section) in self.sections.iter().enumerate() {
            let offset = header_len + 40 * i;
            bytes[offset..offset + section.name.len()].copy_from_slice(section.name.as_bytes());
            put32(&mut bytes, offset + 16, section.data.len() as u32);
            let data_offset = bytes.len() as u32;
            put32(&mut bytes, offset + 20, data_offset);
            let flags = if section.name == ".text" {
                0x6050_0020
            } else {
                0x4030_0040
            };
            put32(
                &mut bytes,
                offset + 36,
                flags | if section.comdat.is_some() { 0x1000 } else { 0 },
            );
            bytes.extend_from_slice(&section.data);
        }
        let symbol_offset = bytes.len() as u32;
        let mut strings = vec![0; 4];
        let mut symbol_bytes = Vec::new();
        for (i, section) in self.sections.iter().enumerate() {
            if let Some(selection) = section.comdat {
                let mut raw = vec![0; symbol_len];
                raw[..section.name.len()].copy_from_slice(section.name.as_bytes());
                write_symbol_fields(&mut raw, self.big, (i + 1) as u32, 3, 1);
                symbol_bytes.extend(raw);
                let mut aux = vec![0; symbol_len];
                put32(&mut aux, 0, section.data.len() as u32);
                aux[14] = selection;
                symbol_bytes.extend(aux);
            }
        }
        for symbol in &self.symbols {
            let mut raw = vec![0; symbol_len];
            if symbol.name.len() <= 8 {
                raw[..symbol.name.len()].copy_from_slice(symbol.name.as_bytes());
            } else {
                put32(&mut raw, 4, strings.len() as u32);
                strings.extend_from_slice(symbol.name.as_bytes());
                strings.push(0);
            }
            write_symbol_fields(&mut raw, self.big, symbol.section, symbol.storage, 0);
            symbol_bytes.extend(raw);
        }
        if self.big {
            put32(&mut bytes, 48, symbol_offset);
            put32(&mut bytes, 52, (symbol_bytes.len() / symbol_len) as u32);
        } else {
            put32(&mut bytes, 8, symbol_offset);
            put32(&mut bytes, 12, (symbol_bytes.len() / symbol_len) as u32);
        }
        let strings_len = strings.len() as u32;
        put32(&mut strings, 0, strings_len);
        bytes.extend(symbol_bytes);
        bytes.extend(strings);
        bytes
    }
}

fn write_symbol_fields(raw: &mut [u8], big: bool, section: u32, storage: u8, aux: u8) {
    if big {
        put32(raw, 12, section);
    } else {
        put16(raw, 12, section as u16);
    }
    let class = if big { 18 } else { 16 };
    raw[class] = storage;
    raw[class + 1] = aux;
}

fn put16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn append_member(archive: &mut Vec<u8>, name: &str, data: &[u8]) {
    let header = format!(
        "{name:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
        0,
        0,
        0,
        0,
        data.len()
    );
    assert_eq!(header.len(), 60);
    archive.extend_from_slice(header.as_bytes());
    archive.extend_from_slice(data);
    if data.len() % 2 != 0 {
        archive.push(b'\n');
    }
}

/// Deterministic COFF archive with both real linker indexes and NUL-terminated
/// MSVC long names. Expectations come from the fixture's declared exports.
fn archive(objects: &[(&str, Vec<u8>, Vec<&str>)]) -> Vec<u8> {
    let mut names = Vec::new();
    let mut name_offsets = Vec::new();
    let mut symbols = Vec::new();
    for (i, (name, _, exports)) in objects.iter().enumerate() {
        name_offsets.push(names.len());
        names.extend_from_slice(name.as_bytes());
        names.push(0);
        for symbol in exports {
            symbols.push(((*symbol).to_owned(), i));
        }
    }
    symbols.sort();
    let symbol_names_size: usize = symbols.iter().map(|(s, _)| s.len() + 1).sum();
    let first_size = 4 + symbols.len() * 4 + symbol_names_size;
    let second_size = 4 + objects.len() * 4 + 4 + symbols.len() * 2 + symbol_names_size;
    let member_size = |n: usize| 60 + n + n % 2;
    let mut offset =
        8 + member_size(first_size) + member_size(second_size) + member_size(names.len());
    let mut offsets = Vec::new();
    for (_, bytes, _) in objects {
        offsets.push(offset as u32);
        offset += member_size(bytes.len());
    }
    let mut first = Vec::new();
    first.extend_from_slice(&(symbols.len() as u32).to_be_bytes());
    for (_, i) in &symbols {
        first.extend_from_slice(&offsets[*i].to_be_bytes());
    }
    for (symbol, _) in &symbols {
        first.extend_from_slice(symbol.as_bytes());
        first.push(0);
    }
    let mut second = Vec::new();
    second.extend_from_slice(&(objects.len() as u32).to_le_bytes());
    for offset in &offsets {
        second.extend_from_slice(&offset.to_le_bytes());
    }
    second.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
    for (_, i) in &symbols {
        second.extend_from_slice(&((*i + 1) as u16).to_le_bytes());
    }
    for (symbol, _) in &symbols {
        second.extend_from_slice(symbol.as_bytes());
        second.push(0);
    }
    let mut result = b"!<arch>\n".to_vec();
    append_member(&mut result, "/", &first);
    append_member(&mut result, "/", &second);
    append_member(&mut result, "//", &names);
    for ((_, bytes, _), name_offset) in objects.iter().zip(name_offsets) {
        append_member(&mut result, &format!("/{name_offset}"), bytes);
    }
    result
}

pub(crate) fn native_library(exports: &[&str]) -> Vec<u8> {
    archive(&[(
        "native_operator_long_member_name.obj",
        ObjectFixture::code(exports).bytes(),
        exports.to_vec(),
    )])
}

#[test]
fn reads_real_coff_and_bigobj_members_names_bytes_and_symbols() {
    let normal = ObjectFixture::code(&["ferrum_execute"]);
    let mut big = ObjectFixture::code(&["ferrum_descriptor"]);
    big.big = true;
    let normal_bytes = normal.bytes();
    let big_bytes = big.bytes();
    let input = archive(&[
        (
            "path/to/a_long_native_member.obj",
            normal_bytes.clone(),
            vec!["ferrum_execute"],
        ),
        (
            "another_big_member.obj",
            big_bytes.clone(),
            vec!["ferrum_descriptor"],
        ),
    ]);
    let inspected = inspect_msvc_archive(&input, TARGET).unwrap();
    assert_eq!(inspected.identity.machine, 0x8664);
    assert_eq!(inspected.identity.class_bits, 64);
    assert_eq!(
        inspected.members[0].name,
        "path/to/a_long_native_member.obj"
    );
    assert_eq!(inspected.members[0].bytes, normal_bytes);
    assert_eq!(inspected.members[1].bytes, big_bytes);
    assert_eq!(
        inspected.members[1].sha256,
        format!("{:x}", Sha256::digest(&big_bytes))
    );
    assert_eq!(
        inspected.strong_defined_symbols,
        ["ferrum_descriptor", "ferrum_execute"]
    );
    inspected
        .require_unique_exports(&["ferrum_execute".into(), "ferrum_descriptor".into()])
        .unwrap();
    assert!(inspected
        .require_unique_exports(&["missing_export".into()])
        .is_err());
}

#[test]
fn undefined_import_references_are_not_definitions_or_import_libraries() {
    let mut native = ObjectFixture::code(&["ferrum_execute"]);
    native.symbols.push(SymbolFixture {
        name: "__imp_cudaMalloc".into(),
        section: 0,
        storage: 2,
    });
    let inspected = inspect_msvc_object(&native.bytes(), TARGET).unwrap();
    assert_eq!(inspected.defined_symbols, ["ferrum_execute"]);
}

#[test]
fn rejects_short_long_and_mixed_import_archives() {
    let mut short = vec![0; 20];
    put16(&mut short, 2, 0xffff);
    put16(&mut short, 6, 0x8664);
    let strings = b"ferrum_execute\0provider.dll\0";
    put32(&mut short, 12, strings.len() as u32);
    short.extend_from_slice(strings);
    let mut long = ObjectFixture::code(&["ferrum_execute"]);
    long.sections.push(SectionFixture {
        name: ".idata$5",
        data: vec![0; 8],
        comdat: None,
    });
    let marker = ObjectFixture::code(&["__IMPORT_DESCRIPTOR_provider"]).bytes();
    for bytes in [short, long.bytes(), marker] {
        assert!(inspect_msvc_object(&bytes, TARGET).is_err());
        assert!(inspect_msvc_archive(
            &archive(&[("import.obj", bytes.clone(), vec!["ferrum_execute"])]),
            TARGET
        )
        .is_err());
        assert!(inspect_msvc_archive(
            &archive(&[
                (
                    "real.obj",
                    ObjectFixture::code(&["real"]).bytes(),
                    vec!["real"]
                ),
                ("import.obj", bytes, vec!["ferrum_execute"])
            ]),
            TARGET
        )
        .is_err());
    }
}

#[test]
fn rejects_thin_truncated_wrong_machine_and_disguised_elf() {
    let input = native_library(&["ferrum_execute"]);
    let mut thin = input.clone();
    thin[..8].copy_from_slice(b"!<thin>\n");
    assert!(inspect_msvc_archive(&thin, TARGET).is_err());
    assert!(inspect_msvc_archive(&input[..input.len() - 3], TARGET).is_err());
    for machine in [0x014c, 0xaa64] {
        let mut wrong = ObjectFixture::code(&["ferrum_execute"]);
        wrong.machine = machine;
        assert!(inspect_msvc_object(&wrong.bytes(), TARGET).is_err());
        wrong.big = true;
        assert!(inspect_msvc_object(&wrong.bytes(), TARGET).is_err());
    }
    let elf = b"\x7fELF\x02\x01\x01\0not-a-coff-object".to_vec();
    assert!(inspect_msvc_archive(
        &archive(&[("spoof.obj", elf, vec!["ferrum_execute"])]),
        TARGET
    )
    .is_err());
    assert!(inspect_msvc_archive(&input, "x86_64-unknown-linux-gnu").is_err());
    assert!(inspect_msvc_archive(&input, "aarch64-pc-windows-msvc").is_err());
    let mut bad_aux = ObjectFixture::code(&["foo"]).bytes();
    let symbols = u32::from_le_bytes(bad_aux[8..12].try_into().unwrap()) as usize;
    bad_aux[symbols + 17] = 1;
    assert!(inspect_msvc_object(&bad_aux, TARGET).is_err());
}

#[test]
fn honors_comdat_selection_but_rejects_duplicate_abi_entrypoints() {
    let mut inline = ObjectFixture::code(&["inline_function"]);
    inline.sections[0].comdat = Some(2);
    let input = archive(&[
        ("first.obj", inline.bytes(), vec!["inline_function"]),
        ("second.obj", inline.bytes(), vec!["inline_function"]),
    ]);
    let inspected = inspect_msvc_archive(&input, TARGET).unwrap();
    assert!(inspected.strong_defined_symbols.is_empty());
    assert_eq!(inspected.symbol_definition_counts["inline_function"], 2);
    assert!(inspected
        .require_unique_exports(&["inline_function".into()])
        .is_err());
    inline.sections[0].comdat = Some(1);
    assert!(inspect_msvc_archive(
        &archive(&[
            ("first.obj", inline.bytes(), vec!["inline_function"]),
            ("second.obj", inline.bytes(), vec!["inline_function"])
        ]),
        TARGET
    )
    .is_err());
    let strong = ObjectFixture::code(&["ferrum_execute"]).bytes();
    assert!(inspect_msvc_archive(
        &archive(&[
            ("a.obj", strong.clone(), vec!["ferrum_execute"]),
            ("b.obj", strong, vec!["ferrum_execute"])
        ]),
        TARGET
    )
    .is_err());
}

#[test]
fn linker_index_is_not_export_proof_and_must_resolve_the_real_member() {
    let input = archive(&[(
        "first.obj",
        ObjectFixture::code(&["real"]).bytes(),
        vec!["invented"],
    )]);
    let inspected = inspect_msvc_archive(&input, TARGET).unwrap();
    assert!(inspected
        .require_unique_exports(&["invented".into()])
        .is_err());
    assert!(inspected.require_unique_exports(&["real".into()]).is_err());
    let swapped = archive(&[
        (
            "first.obj",
            ObjectFixture::code(&["real"]).bytes(),
            vec!["other"],
        ),
        (
            "second.obj",
            ObjectFixture::code(&["other"]).bytes(),
            vec!["real"],
        ),
    ]);
    assert!(inspect_msvc_archive(&swapped, TARGET)
        .unwrap()
        .require_unique_exports(&["real".into()])
        .is_err());
}

#[test]
fn release_dynamic_crt_rejects_static_and_debug_linker_directives() {
    for directive in [
        " /DEFAULTLIB:MSVCRT /DEFAULTLIB:MSVCPRT",
        " /DEFAULTLIB:ucrt /DEFAULTLIB:vcruntime",
        " /FAILIFMISMATCH:\"RuntimeLibrary=MD_DynamicRelease\"",
    ] {
        let mut fixture = ObjectFixture::code(&["ferrum_execute"]);
        fixture.sections.push(SectionFixture {
            name: ".drectve",
            data: directive.as_bytes().to_vec(),
            comdat: None,
        });
        inspect_msvc_object(&fixture.bytes(), TARGET).unwrap();
    }
    for directive in [
        " /DEFAULTLIB:LIBCMT",
        " /DEFAULTLIB:LIBVCRUNTIME /DEFAULTLIB:LIBUCRT",
        " /DEFAULTLIB:\"C:\\Program Files\\MSVC\\libcmt.lib\"",
        " /DEFAULTLIB:\"libcpmt.lib\"",
        " -defaultlib:msvcrtd",
        " /FAILIFMISMATCH:\"RuntimeLibrary=MT_StaticRelease\"",
        " /FAILIFMISMATCH:\"RuntimeLibrary=MD_DynamicRelease_extra\"",
    ] {
        let mut fixture = ObjectFixture::code(&["ferrum_execute"]);
        fixture.sections.push(SectionFixture {
            name: ".drectve",
            data: directive.as_bytes().to_vec(),
            comdat: None,
        });
        assert!(
            inspect_msvc_object(&fixture.bytes(), TARGET).is_err(),
            "{directive}"
        );
    }
}
