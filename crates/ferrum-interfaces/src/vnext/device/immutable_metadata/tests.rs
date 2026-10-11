use super::*;
use crate::vnext::{BufferUsage, ElementType};
use std::sync::atomic::Ordering;

#[path = "../../../../tests/vnext_device_operation_contract/mod.rs"]
mod operation_fixture;
use operation_fixture::{catalog, id, runtime, TestBuffer};

fn buffer() -> TestBuffer {
    TestBuffer {
        descriptor: BufferDescriptor {
            resource_id: id("resource.metadata-owner"),
            size_bytes: 32,
            alignment_bytes: 8,
            usage: BufferUsage::State,
            element_type: ElementType::U8,
        },
    }
}

#[test]
fn immutable_runtime_declaration_requires_the_exact_runtime_owner() {
    let catalog = catalog();
    let (owner, _) = runtime(&catalog);
    let (donor, _) = runtime(&catalog);
    let alias = Arc::clone(&owner);
    assert_eq!(owner.descriptor(), donor.descriptor());

    let declaration = ImmutableRuntimeMetadata::declare(owner.as_ref(), owner.descriptor());
    assert_eq!(
        declaration.for_owner(alias.as_ref()),
        Some(owner.descriptor())
    );
    assert!(declaration.for_owner(donor.as_ref()).is_none());
}

#[test]
fn immutable_buffer_declaration_rejects_equal_descriptor_donor_facades() {
    let catalog = catalog();
    let (owner, _) = runtime(&catalog);
    let (donor_runtime, _) = runtime(&catalog);
    let owner_buffer = Arc::new(buffer());
    let alias = Arc::clone(&owner_buffer);
    let donor_buffer = TestBuffer {
        descriptor: owner_buffer.descriptor.clone(),
    };
    let declaration = ImmutableBufferMetadata::declare(
        owner.as_ref(),
        owner_buffer.as_ref(),
        &owner_buffer.descriptor,
    );

    assert_eq!(
        declaration.for_owner(owner.as_ref(), alias.as_ref()),
        Some(&owner_buffer.descriptor)
    );
    assert!(declaration
        .for_owner(owner.as_ref(), &donor_buffer)
        .is_none());
    assert!(declaration
        .for_owner(donor_runtime.as_ref(), owner_buffer.as_ref())
        .is_none());
}

#[test]
fn default_metadata_capabilities_do_not_hide_dynamic_getter_values() {
    let (runtime, trace) = runtime(&catalog());
    let buffer = buffer();
    assert!(runtime.immutable_runtime_metadata().is_none());
    assert!(runtime.immutable_buffer_metadata(&buffer).is_none());
    let initial_runtime = runtime.descriptor().clone();
    let initial_buffer = runtime.buffer_descriptor(&buffer);

    runtime
        .use_alternate_descriptor
        .store(true, Ordering::Release);
    trace.lock().unwrap().tamper_buffer_descriptor = true;

    assert!(runtime.immutable_runtime_metadata().is_none());
    assert!(runtime.immutable_buffer_metadata(&buffer).is_none());
    assert_eq!(runtime.descriptor().id, initial_runtime.id);
    assert_ne!(runtime.descriptor(), &initial_runtime);
    let changed_buffer = runtime.buffer_descriptor(&buffer);
    assert_eq!(changed_buffer.resource_id, initial_buffer.resource_id);
    assert_eq!(changed_buffer.size_bytes, initial_buffer.size_bytes + 1);
}

#[test]
fn entry_markers_keep_clone_identity_without_aliasing_a_replacement() {
    let original = DeviceReusableExecutionEntryIdentity::new();
    let retained = original.clone();
    assert!(original.same_entry(&retained));
    drop(original);

    let replacement = DeviceReusableExecutionEntryIdentity::new();
    assert!(!retained.same_entry(&replacement));
    assert!(replacement.same_entry(&replacement.clone()));
}
