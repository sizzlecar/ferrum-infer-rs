use super::*;

fn request() -> CutlassNvfp4RequestV1 {
    CutlassNvfp4RequestV1::new(
        129,
        128,
        136,
        CutlassNvfp4Device {
            ordinal: 0,
            architecture: 1200,
            multiprocessors: 170,
            maximum_dynamic_shared_bytes: 100_000,
        },
    )
    .unwrap()
}
fn bindings(plan: &CutlassNvfp4PlanV1) -> CutlassNvfp4BindingsV1 {
    let span = |address, bytes| CutlassNvfp4Span { address, bytes };
    CutlassNvfp4BindingsV1 {
        input: span(0x10000, plan.request.input_bytes().unwrap()),
        weight: span(0x20000, plan.request.weight_bytes().unwrap()),
        input_scales: span(0x30000, plan.request.input_scale_bytes().unwrap()),
        weight_scales: span(0x40000, plan.request.weight_scale_bytes().unwrap()),
        alpha: span(0x50000, 4),
        output: span(0x60000, plan.request.output_bytes().unwrap()),
        workspace: if plan.workspace_bytes == 0 {
            CutlassNvfp4Span::default()
        } else {
            span(0x80000, plan.workspace_bytes)
        },
    }
}
#[test]
fn packed_geometry_checks_padding_and_backend_restrictions() {
    let r = request();
    assert_eq!(r.input_bytes().unwrap(), 8256);
    assert_eq!(r.input_scale_bytes().unwrap(), 2048);
    assert_eq!(r.weight_scale_bytes().unwrap(), 2048);
    assert_eq!(r.output_bytes().unwrap(), 35088);
    for (inputs, outputs, architecture) in [(96, 136, 1200), (128, 135, 1200), (128, 136, 1000)] {
        let mut wrong = r;
        wrong.inputs = inputs;
        wrong.outputs = outputs;
        wrong.architecture = architecture;
        assert_eq!(
            wrong.validate(),
            Err(CutlassNvfp4Error::UnsupportedGeometry)
        );
    }
    let mut wrong = r;
    wrong.rows = u32::MAX;
    assert_eq!(wrong.validate(), Err(CutlassNvfp4Error::InvalidArgument));
}
#[test]
fn spans_reject_each_read_write_overlap_and_bad_extent_before_enqueue() {
    let plan = CutlassNvfp4PlanV1::checked(request(), 65536, 1024).unwrap();
    let valid = bindings(&plan);
    valid.validate(&plan).unwrap();
    for read in [
        valid.input,
        valid.weight,
        valid.input_scales,
        valid.weight_scales,
        valid.alpha,
    ] {
        let mut wrong = valid;
        wrong.output.address = read.address;
        assert_eq!(wrong.validate(&plan), Err(CutlassNvfp4Error::Span));
        wrong = valid;
        wrong.workspace.address = read.address;
        assert_eq!(wrong.validate(&plan), Err(CutlassNvfp4Error::Span));
    }
    let mut wrong = valid;
    wrong.weight_scales.bytes -= 1;
    assert_eq!(wrong.validate(&plan), Err(CutlassNvfp4Error::Span));
    wrong = valid;
    wrong.input.address = u64::MAX - 15;
    assert_eq!(wrong.validate(&plan), Err(CutlassNvfp4Error::Span));
    wrong = valid;
    wrong.workspace.address += 16;
    assert_eq!(wrong.validate(&plan), Err(CutlassNvfp4Error::Span));
    wrong = valid;
    wrong.workspace.address = wrong.output.address;
    assert_eq!(wrong.validate(&plan), Err(CutlassNvfp4Error::Span));
}
#[test]
fn zero_workspace_is_explicit_and_plan_tampering_fails() {
    let plan = CutlassNvfp4PlanV1::checked(request(), 65536, 0).unwrap();
    let mut b = bindings(&plan);
    b.validate(&plan).unwrap();
    b.workspace.address = 0x80000;
    assert_eq!(b.validate(&plan), Err(CutlassNvfp4Error::Span));
    let mut changed = plan;
    changed.kernel_id = 2;
    assert_eq!(changed.validate(), Err(CutlassNvfp4Error::Plan));
    assert_ne!(plan.fingerprint_material(), changed.fingerprint_material());
    assert!(CutlassNvfp4PlanV1::checked(request(), 100_001, 0).is_err());
    assert!(CutlassNvfp4PlanV1::checked(request(), 65536, u64::MAX).is_err());
}
