use super::*;

#[path = "../../../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use vnext_device_operation_contract::{fixture_with_device_id, id};
use vnext_device_operation_wave_contract::{setup_with_fixture, teardown};

static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn fixture() -> vnext_device_operation_contract::Fixture {
    fixture_with_device_id(id(format!(
        "device.prepared-core-io.{}",
        NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )))
}

fn rows() -> [OperationCostWorkRow; 2] {
    std::array::from_fn(|_| OperationCostWorkRow {
        offset: 0,
        count: std::num::NonZeroU64::new(1).unwrap(),
        full_input_tokens: std::num::NonZeroU64::new(1).unwrap(),
    })
}

#[test]
fn prepared_core_io_index_preserves_dispatch_order_and_rejects_foreign_plan() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture());
    let (foreign, foreign_sequence, foreign_session, foreign_batch, foreign_step) =
        setup_with_fixture(fixture());
    {
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        let nodes = f.plan.payload().nodes();
        assert!(nodes.len() > 1);
        for node in nodes {
            let selected = providers
                .prepared_node(&f.resolved, node.id())
                .unwrap()
                .unwrap();
            assert_eq!(selected, node);
            assert!(std::ptr::eq(
                selected,
                f.resolved
                    .execution_plan()
                    .payload()
                    .nodes()
                    .iter()
                    .find(|candidate| candidate.id() == node.id())
                    .unwrap()
            ));
            assert!(providers
                .prepared_node(&foreign.resolved, node.id())
                .is_err());
        }
        assert!(providers
            .prepared_node(&f.resolved, &id("node.absent"))
            .unwrap()
            .is_none());
        // The index is separate from the physical dispatch order.
        assert_eq!(providers.len(), nodes.len());
        for (provider, node) in providers.providers().iter().zip(nodes) {
            provider.validate_binding(&f.resolved, node.id()).unwrap();
        }
    }
    teardown(
        foreign,
        foreign_sequence,
        foreign_session,
        foreign_batch,
        foreign_step,
    );
    teardown(f, sequence, session, batch, step);
}

#[test]
fn prepared_core_io_readbacks_keep_each_row_range_layout_and_budget_checks() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        let work = rows();
        let rows = ValidatedCostRows::new(&work).unwrap();
        let node = id("node.tail");
        let resource = id("resource.output");
        let mut readbacks = [
            EagerCoreReadback {
                node_id: &node,
                resource_id: &resource,
                participant_index: 0,
                logical_offset_bytes: 0,
                layout: HostTransferLayout::new(ElementType::F32, 1).unwrap(),
            },
            EagerCoreReadback {
                node_id: &node,
                resource_id: &resource,
                participant_index: 1,
                logical_offset_bytes: 0,
                layout: HostTransferLayout::new(ElementType::F32, 2).unwrap(),
            },
        ];
        let query = |readbacks: &[EagerCoreReadback<'_>]| {
            readback_bytes(&providers, &f.resolved, &rows, readbacks, &mut || true)
        };
        assert_eq!(query(&readbacks), Ok(12));
        readbacks[1].participant_index = 2;
        assert_eq!(query(&readbacks), Err(U::InvalidInput));
        readbacks[1].participant_index = 1;
        readbacks[1].logical_offset_bytes = u64::MAX;
        assert_eq!(query(&readbacks), Err(U::InvalidInput));
        readbacks[1].logical_offset_bytes = 0;
        readbacks[1].layout = HostTransferLayout::new(ElementType::U32, 2).unwrap();
        assert_eq!(query(&readbacks), Err(U::InvalidInput));
        readbacks[1].layout = HostTransferLayout::new(ElementType::F32, 2).unwrap();
        let absent = id("node.absent");
        readbacks[1].node_id = &absent;
        assert_eq!(query(&readbacks), Err(U::InvalidInput));
        readbacks[1].node_id = &node;
        assert_eq!(
            readback_bytes(&providers, &f.resolved, &rows, &readbacks, &mut || false),
            Err(U::BudgetExhausted)
        );
    }
    teardown(f, sequence, session, batch, step);
}

#[test]
fn prepared_core_io_uploads_keep_group_order_and_per_participant_validation() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        let work = rows();
        let rows = ValidatedCostRows::new(&work).unwrap();
        let (node, value) = f
            .plan
            .payload()
            .nodes()
            .iter()
            .find_map(|node| {
                node.values()
                    .iter()
                    .find(|value| {
                        value.role() == ResolvedValueRole::Input
                            && value.usage() == BufferUsage::Activations
                            && value.storage().components().len() == 1
                            && contiguous_step_descriptor(
                                &f.resolved,
                                value.storage().components()[0].resource_id(),
                            )
                            .is_ok()
                    })
                    .map(|value| (node, value))
            })
            .expect("fixture has a contiguous Step activation input");
        let element = value.tensor().element_type();
        let layout = HostTransferLayout::new(element, 1).unwrap();
        let mut uploads = std::array::from_fn::<_, 2, _>(|index| EagerCoreInputUpload {
            node_id: node.id(),
            input_ordinal: value.ordinal(),
            participant_index: index,
            logical_offset_bytes: 0,
            layout,
        });
        let query = |uploads: &[EagerCoreInputUpload<'_>]| {
            input_transfer_bytes(&providers, &f.resolved, &rows, uploads, &mut || true)
        };
        assert_eq!(
            query(&uploads).unwrap().iter().sum::<u64>(),
            2 * layout.byte_len().unwrap()
        );
        uploads[1].participant_index = 2;
        assert_eq!(query(&uploads), Err(U::InvalidInput));
        uploads[1].participant_index = 1;
        uploads[1].logical_offset_bytes = u64::MAX;
        assert_eq!(query(&uploads), Err(U::InvalidInput));
        uploads[1].logical_offset_bytes = 0;
        let absent = id("node.absent");
        uploads[1].node_id = &absent;
        assert_eq!(query(&uploads), Err(U::InvalidInput));
        uploads[1].node_id = node.id();
        assert_eq!(
            input_transfer_bytes(&providers, &f.resolved, &rows, &uploads, &mut || false),
            Err(U::BudgetExhausted)
        );
    }
    teardown(f, sequence, session, batch, step);
}
