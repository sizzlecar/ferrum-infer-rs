use super::super::tests::{archive, config, limits, original, prompt};
use super::*;
fn account(limit: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            limit.max(1),
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: limit,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
#[test]
fn original_bf16_norms_promote_under_actual_weightbacking_and_refund() {
    let c = config(prompt().layout().clone());
    let (_directory, source) = archive(&original(&c));
    let host = account(1 << 20, 64);
    let adapter = PH3ImageTextWeightAdapter::new(&c, Arc::new(source), limits()).unwrap();
    let wrapper = adapter.weight_source(&host);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let norm = adapter
        .identity
        .schema()
        .components
        .iter()
        .find(|c| adapter.promotions.contains(&c.id) && c.dimensions == [256])
        .unwrap();
    let bytes = wrapper.component_backing_bytes(norm).unwrap();
    assert_eq!(bytes, host.backing_bytes(256 * 4).unwrap());
    let payload = wrapper.component(norm).unwrap();
    assert_eq!(payload.element_type(), ElementType::F32);
    assert!(payload.retained_host_memory().is_some());
    assert_eq!(host.usage().unwrap().committed_bytes(), bytes);
    assert!(payload
        .bytes()
        .chunks_exact(4)
        .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 1.));
    assert_ne!(
        payload.bytes().as_ptr(),
        adapter
            .source
            .archive()
            .tensor(&norm.external_names[0])
            .unwrap()
            .bytes()
            .as_ptr()
    );
    drop(payload);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let too_small = account(bytes - 1, 64);
    let denied = adapter.weight_source(&too_small);
    assert!(denied.component(norm).is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
    let mut foreign = norm.clone();
    foreign.dimensions[0] += 1;
    assert!(wrapper.component(&foreign).is_err());
    foreign = norm.clone();
    foreign.encoding = WeightEncoding::Dense {
        element_type: ElementType::Bf16,
    };
    assert!(wrapper.component_backing_bytes(&foreign).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn quantized_weights_and_original_f8_carriers_remain_borrowed_and_uncharged() {
    let c = config(prompt().layout().clone());
    let (_directory, source) = archive(&original(&c));
    let host = account(0, 64);
    let adapter = PH3ImageTextWeightAdapter::new(&c, Arc::new(source), limits()).unwrap();
    let wrapper = adapter.weight_source(&host);
    let mut carrier = false;
    let mut packed = false;
    for component in adapter
        .identity
        .schema()
        .components
        .iter()
        .filter(|c| !adapter.promotions.contains(&c.id))
    {
        assert_eq!(wrapper.component_backing_bytes(component).unwrap(), 0);
        let payload = wrapper.component(component).unwrap();
        let raw = adapter
            .source
            .archive()
            .tensor(&component.external_names[0])
            .unwrap();
        assert_eq!(payload.bytes().as_ptr(), raw.bytes().as_ptr());
        assert!(payload.retained_host_memory().is_some());
        if component.external_names[0].ends_with(".weight_s_rel") {
            carrier = true;
            assert_eq!(raw.dtype(), safetensors::Dtype::F8_E4M3);
            assert_eq!(payload.element_type(), ElementType::U8);
            assert!(payload.bytes().iter().all(|&b| b == 0x38));
        }
        if payload.element_type() == ElementType::I8 {
            packed = true;
        }
    }
    assert!(carrier && packed);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn static_group_peak_is_alignment_aware_placement_sensitive_and_not_schema_sum() {
    let c = config(prompt().layout().clone());
    let (_directory, source) = archive(&original(&c));
    let host = account(1 << 20, 64);
    let adapter = PH3ImageTextWeightAdapter::new(&c, Arc::new(source), limits()).unwrap();
    let all = adapter
        .identity
        .schema()
        .components
        .iter()
        .map(|c| c.id.clone())
        .collect::<BTreeSet<_>>();
    let norm_sum = adapter
        .identity
        .schema()
        .components
        .iter()
        .map(|c| adapter.component_backing_bytes(c, &host).unwrap())
        .sum::<usize>();
    let expected = host.backing_bytes(256 * 4).unwrap();
    assert_eq!(adapter.group_peak(&all, &host).unwrap(), expected);
    assert!(norm_sum > expected);
    let head = adapter
        .identity
        .schema()
        .components
        .iter()
        .find(|c| adapter.promotions.contains(&c.id) && c.dimensions == [128])
        .unwrap();
    let carrier = adapter
        .identity
        .schema()
        .components
        .iter()
        .find(|c| c.external_names[0].ends_with(".weight_s_rel"))
        .unwrap();
    assert_eq!(
        adapter
            .group_peak(
                &BTreeSet::from([head.id.clone(), carrier.id.clone()]),
                &host
            )
            .unwrap(),
        host.backing_bytes(128 * 4).unwrap()
    );
    assert_eq!(
        adapter
            .group_peak(&BTreeSet::from([carrier.id.clone()]), &host)
            .unwrap(),
        0
    );
    assert!(adapter
        .group_peak(
            &BTreeSet::from([WeightId::new("component.foreign").unwrap()]),
            &host
        )
        .is_err());
    let mut unknown = serde_json::to_value(&adapter.identity).unwrap();
    unknown["materializer_id"] = serde_json::json!("materializer.unapproved.text");
    let rejected = serde_json::from_value::<ExecutionWeightPlan>(unknown)
        .map(|p| adapter.check_identity(&p).is_err())
        .unwrap_or(true);
    assert!(rejected);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
