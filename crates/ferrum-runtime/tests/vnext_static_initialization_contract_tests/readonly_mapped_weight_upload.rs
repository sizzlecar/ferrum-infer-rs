//! Real read-only file mappings through static admission and original fences.
use super::*;
use std::io::Write;

struct Mapping {
    bytes: Option<memmap2::Mmap>,
    path: std::path::PathBuf,
    drops: Arc<AtomicU64>,
}
impl Mapping {
    fn new(drops: Arc<AtomicU64>) -> Arc<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrum-readonly-static-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        writer.write_all(&(0u8..64).collect::<Vec<_>>()).unwrap();
        drop(writer);
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: this fixture never writes or truncates the file. Only the
        // private read-only mapping is retained, and cleanup unmaps it first.
        let bytes = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
        Arc::new(Self {
            bytes: Some(bytes),
            path,
            drops,
        })
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        drop(self.bytes.take());
        let _ = std::fs::remove_file(&self.path);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
// SAFETY: the private read-only file map is immutable until the last Arc drops.
unsafe impl StableHostMemory for Mapping {
    fn stable_bytes(&self) -> &[u8] {
        self.bytes.as_ref().unwrap()
    }
}
// SAFETY: unlike an ordinary stable Vec, Mapping really owns a read-only file
// mapping. Its Drop unmaps only after all retained regions/fences disappear.
unsafe impl ImmutableFileMappedHostMemory for Mapping {}

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            15,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 15,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        8,
    )
    .unwrap()
}
fn enable(trace: &Arc<Mutex<RuntimeTrace>>, mapped: bool) {
    let mut t = trace.lock().unwrap();
    t.managed_static_uploads_enabled = true;
    t.readonly_mapped_static_uploads_enabled = mapped;
}
fn source_range(c: &WeightComponentSpec) -> std::ops::Range<usize> {
    if c.id.as_str().ends_with("left") {
        8..16
    } else {
        32..40
    }
}

struct Source {
    owner: Arc<Mapping>,
    trace: Arc<Mutex<RuntimeTrace>>,
    calls: Mutex<Vec<u64>>,
    ordinary_right: bool,
    indeterminate_once: AtomicBool,
}
impl Source {
    fn new(owner: Arc<Mapping>, trace: Arc<Mutex<RuntimeTrace>>) -> Self {
        Self {
            owner,
            trace,
            calls: Mutex::new(Vec::new()),
            ordinary_right: false,
            indeterminate_once: AtomicBool::new(false),
        }
    }
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.calls
            .lock()
            .unwrap()
            .push(self.trace.lock().unwrap().wait_fence_calls);
        if self.indeterminate_once.swap(false, Ordering::SeqCst) {
            self.trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
        }
        let range = source_range(c);
        let bytes = &self.owner.stable_bytes()[range.clone()];
        if self.ordinary_right && c.id.as_str().ends_with("right") {
            return WeightComponentPayload::new(
                c,
                c.external_names[0].clone(),
                "weights.safetensors",
                c.dimensions.clone(),
                c.physical_element_type(),
                bytes.to_vec(),
            );
        }
        let retained = RetainedHostMemoryRegion::new_readonly_file_mapping(
            Arc::clone(&self.owner),
            range.start,
            range.len(),
        )?;
        WeightComponentPayload::new(
            c,
            c.external_names[0].clone(),
            "weights.safetensors",
            c.dimensions.clone(),
            c.physical_element_type(),
            bytes,
        )?
        .with_retained_host_memory(retained)
    }
}

#[test]
fn readonly_mapped_upload_chunks_keep_exact_nonzero_source_and_destination_offsets() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let drops = Arc::new(AtomicU64::new(0));
    let source = Source::new(Mapping::new(Arc::clone(&drops)), Arc::clone(&trace));
    let account = host();
    let initialized = committed_transaction(&plan, runtime, "mapped-chunks")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(4, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 16);
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 0);
    assert_eq!(initialized.receipt().staging_copy_duration_us(), 0);
    assert_eq!(
        trace.lock().unwrap().mapped_upload_ranges,
        vec![
            (8, 0, (8u8..12).collect::<Vec<_>>()),
            (12, 4, (12u8..16).collect::<Vec<_>>()),
            // The second component has its own static resource, so its
            // destination offsets restart at zero rather than following left.
            (32, 0, (32u8..36).collect::<Vec<_>>()),
            (36, 4, (36u8..40).collect::<Vec<_>>()),
        ]
    );
    assert!(trace.lock().unwrap().managed_upload_classes.is_empty());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close(handoff(initialized));
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn readonly_mapped_upload_finishes_each_source_group_before_next_materialization() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let source = Source::new(
        Mapping::new(Arc::new(AtomicU64::new(0))),
        Arc::clone(&trace),
    );
    let initialized = committed_transaction(&plan, runtime, "mapped-group-boundary")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &host(),
            StaticInitializationPolicy::new(8, 64).unwrap(),
        )
        .unwrap();
    let calls = source.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(
        calls[1] > calls[0],
        "no fence was observed before the next source group"
    );
    drop(calls);
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 16);
    close(handoff(initialized));
}

struct UnknownBytes(Vec<u8>);
// SAFETY: immutable owned bytes are stable, but are deliberately NOT a mapping.
unsafe impl StableHostMemory for UnknownBytes {
    fn stable_bytes(&self) -> &[u8] {
        &self.0
    }
}
struct UnknownSource(Arc<UnknownBytes>);
impl WeightComponentSource for UnknownSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let range = source_range(c);
        let retained =
            RetainedHostMemoryRegion::new(Arc::clone(&self.0), range.start, range.len())?;
        assert!(retained.managed_memory_class().is_none());
        assert!(retained.readonly_file_mapping().is_none());
        WeightComponentPayload::new(
            c,
            c.external_names[0].clone(),
            "weights.safetensors",
            c.dimensions.clone(),
            c.physical_element_type(),
            &self.0 .0[range],
        )?
        .with_retained_host_memory(retained)
    }
}

#[test]
fn unknown_vec_retained_region_never_uses_mapped_upload() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let source = UnknownSource(Arc::new(UnknownBytes((0u8..64).collect())));
    let account = host();
    let initialized = committed_transaction(&plan, runtime, "mapped-unknown-vec")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 0);
    assert_eq!(trace.lock().unwrap().mapped_upload_encode_calls, 0);
    assert_eq!(
        trace.lock().unwrap().managed_upload_classes,
        vec![HostMemoryClass::TransferStaging; 2]
    );
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close(handoff(initialized));
}

struct ForeignManagedSource(HostMemoryAccount);
impl WeightComponentSource for ForeignManagedSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let owner = self
            .0
            .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
            .map_err(|e| VNextError::InvalidExecutionPlan {
                reason: e.to_string(),
            })?
            .freeze();
        assert!(owner.retained_region()?.readonly_file_mapping().is_none());
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec!["weights.safetensors".into()],
            c.dimensions.clone(),
            c.physical_element_type(),
            owner,
        )
    }
}

#[test]
fn mapped_capability_cannot_launder_foreign_managed_weight_authority() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    trace.lock().unwrap().managed_static_weight_sources_enabled = true;
    let foreign = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 8,
            },
        )
        .unwrap(),
    );
    let source = ForeignManagedSource(foreign.clone());
    let initialized = committed_transaction(&plan, runtime, "mapped-foreign-managed")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &host(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 0);
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 0);
    assert_eq!(trace.lock().unwrap().mapped_upload_encode_calls, 0);
    assert_eq!(
        trace.lock().unwrap().managed_upload_classes,
        vec![HostMemoryClass::TransferStaging; 2]
    );
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    close(handoff(initialized));
}

#[test]
fn unsupported_mapped_upload_preserves_original_staging_route() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, false);
    let source = Source::new(
        Mapping::new(Arc::new(AtomicU64::new(0))),
        Arc::clone(&trace),
    );
    let initialized = committed_transaction(&plan, runtime, "mapped-unsupported")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &host(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 0);
    assert_eq!(trace.lock().unwrap().mapped_upload_encode_calls, 0);
    assert_eq!(
        trace.lock().unwrap().managed_upload_classes,
        vec![HostMemoryClass::TransferStaging; 2]
    );
    close(handoff(initialized));
}

#[test]
fn mapped_then_ordinary_payload_keeps_three_routes_separate() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let mut source = Source::new(
        Mapping::new(Arc::new(AtomicU64::new(0))),
        Arc::clone(&trace),
    );
    source.ordinary_right = true;
    let account = host();
    let initialized = committed_transaction(&plan, runtime, "mapped-mixed")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(4, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().mapped_source_upload_bytes(), 8);
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 0);
    assert_eq!(trace.lock().unwrap().mapped_upload_encode_calls, 2);
    assert_eq!(
        trace.lock().unwrap().managed_upload_classes,
        vec![HostMemoryClass::TransferStaging; 2]
    );
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close(handoff(initialized));
}

#[test]
fn indeterminate_mapped_fence_keeps_file_alive_after_source_drop_until_recovery() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let drops = Arc::new(AtomicU64::new(0));
    let source = Source::new(Mapping::new(Arc::clone(&drops)), Arc::clone(&trace));
    source.indeterminate_once.store(true, Ordering::SeqCst);
    let failure = match committed_transaction(&plan, runtime, "mapped-unknown-fence")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &host(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        ) {
        Ok(_) => panic!("indeterminate mapped upload must not initialize"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    trace.lock().unwrap().synchronize_fails = true;
    let failure = match failure.recover() {
        Ok(_) => panic!("failed recovery cannot release the mapped owner"),
        Err(failure) => failure,
    };
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    trace.lock().unwrap().synchronize_fails = false;
    let transaction = failure
        .recover()
        .unwrap_or_else(|_| panic!("quiescent recovery failed"));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(transaction);
}

#[test]
fn declared_mapped_hook_decline_cannot_fall_back_to_raw_or_staging() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    trace.lock().unwrap().decline_readonly_mapped_upload = true;
    let source = Source::new(
        Mapping::new(Arc::new(AtomicU64::new(0))),
        Arc::clone(&trace),
    );
    let account = host();
    let failure = match committed_transaction(&plan, runtime, "mapped-declined")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        ) {
        Ok(_) => panic!("a declared hook decline must remain a contract failure"),
        Err(failure) => failure,
    };
    assert!(!failure.is_indeterminate());
    assert!(failure
        .failure()
        .message()
        .contains("declined declared read-only mapped"));
    assert!(trace.lock().unwrap().uploaded_ranges.is_empty());
    assert!(trace.lock().unwrap().managed_upload_classes.is_empty());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
