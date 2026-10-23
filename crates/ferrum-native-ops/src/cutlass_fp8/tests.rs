use super::*;

fn request() -> CutlassFp8RequestV1 {
    CutlassFp8RequestV1::new(
        3,
        64,
        16,
        CutlassFp8Device {
            ordinal: 0,
            architecture: 1200,
            multiprocessors: 1,
            maximum_dynamic_shared_bytes: 128 * 1024,
        },
    )
    .unwrap()
}
fn plan() -> CutlassFp8PlanV1 {
    CutlassFp8PlanV1::checked(request(), 65536, 1024).unwrap()
}
fn bindings(p: &CutlassFp8PlanV1) -> CutlassFp8BindingsV1 {
    CutlassFp8BindingsV1 {
        input: CutlassFp8Span {
            address: 0x10000,
            bytes: p.request.input_bytes().unwrap(),
        },
        weight: CutlassFp8Span {
            address: 0x20000,
            bytes: p.request.weight_bytes().unwrap(),
        },
        weight_scale: CutlassFp8Span {
            address: 0x30000,
            bytes: 4,
        },
        output: CutlassFp8Span {
            address: 0x40000,
            bytes: p.request.output_bytes().unwrap(),
        },
        scratch: CutlassFp8Span {
            address: 0x50000,
            bytes: p.scratch_bytes,
        },
    }
}

#[test]
fn scratch_regions_are_checked_aligned_and_not_matrix_output_temporaries() {
    let p = plan();
    assert_eq!(p.quantized_activation_bytes, 192);
    assert_eq!(p.activation_scale_offset, 256);
    assert_eq!(p.activation_scale_bytes, 12);
    assert_eq!(p.gemm_workspace_offset, 512);
    assert_eq!(p.scratch_bytes, 1536);
    p.validate().unwrap();
    let no_workspace = CutlassFp8PlanV1::checked(request(), 65536, 0).unwrap();
    assert_eq!(no_workspace.scratch_bytes, 512);
    assert!(CutlassFp8PlanV1::checked(request(), 65536, isize::MAX as u64).is_err());
}

#[test]
fn dimensions_and_device_are_rejected_before_any_payload_exists() {
    let mutations: [fn(&mut CutlassFp8RequestV1); 7] = [
        |r: &mut CutlassFp8RequestV1| r.rows = 0,
        |r: &mut CutlassFp8RequestV1| r.inputs = 17,
        |r: &mut CutlassFp8RequestV1| r.outputs = 15,
        |r: &mut CutlassFp8RequestV1| r.architecture = 1210,
        |r: &mut CutlassFp8RequestV1| r.multiprocessors = 0,
        |r: &mut CutlassFp8RequestV1| r.device_ordinal = u32::MAX,
        |r: &mut CutlassFp8RequestV1| r.maximum_dynamic_shared_bytes = 0,
    ];
    for mutate in mutations {
        let mut r = request();
        mutate(&mut r);
        assert!(r.validate().is_err());
    }
    assert!(CutlassFp8PlanV1::checked(request(), 131073, 0).is_err());
    let mut r = request();
    r.rows = i32::MAX as u32;
    r.inputs = i32::MAX as u32 - 15;
    r.outputs = r.inputs;
    // Geometry can exceed any practical card without wrapping; only real
    // admission authorizes storage. Metadata never allocates those payloads.
    let result = r.validate();
    if usize::BITS == 64 {
        assert!(result.is_ok());
    } else {
        assert!(result.is_err());
    }
}

#[test]
fn binding_extents_aliases_and_offsets_are_strict() {
    let p = plan();
    let b = bindings(&p);
    b.validate(&p).unwrap();
    let mut wrong = b;
    wrong.weight.bytes += 1;
    assert!(wrong.validate(&p).is_err());
    let mut wrong = b;
    wrong.scratch.address += 16;
    assert!(wrong.validate(&p).is_err());
    let mut wrong = b;
    wrong.output.address = b.input.address + 16;
    assert!(wrong.validate(&p).is_err());
    let mut wrong = b;
    wrong.scratch.address = b.weight.address;
    assert!(wrong.validate(&p).is_err());
    let mut overflow = b;
    overflow.output.address = u64::MAX - 15;
    assert!(overflow.validate(&p).is_err());
    let mut same_read = b;
    same_read.weight_scale.address = same_read.input.address;
    same_read.validate(&p).unwrap(); // Read-only alias is safe at this raw boundary.
    let mut adjacent = b;
    adjacent.output.address = adjacent.input.address + adjacent.input.bytes;
    adjacent.validate(&p).unwrap();
}

#[test]
fn prepared_facts_reject_forgery_and_identify_workspace_and_device() {
    let p = plan();
    let mut bad = p;
    bad.activation_scale_offset += 256;
    assert!(bad.validate().is_err());
    let mut bad = p;
    bad.kernel_id += 1;
    assert!(bad.validate().is_err());
    let mut other = p.request;
    other.device_ordinal = 1;
    let q = CutlassFp8PlanV1::checked(other, p.shared_bytes, p.gemm_workspace_bytes).unwrap();
    assert_ne!(p.fingerprint_material(), q.fingerprint_material());
    let q = CutlassFp8PlanV1::checked(p.request, p.shared_bytes, 2048).unwrap();
    assert_ne!(p.fingerprint_material(), q.fingerprint_material());
    assert_eq!(p.fingerprint_material()[0..4], 1u32.to_le_bytes());
}
