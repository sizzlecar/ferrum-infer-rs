//! Error-only numeric inventory evidence. This never reads device state or
//! creates a catalog, and cancellation never invokes it.
use super::*;
use ferrum_interfaces::vnext::DeviceCostGraphCatalogLimits;

#[derive(Debug, PartialEq, Eq)]
struct CatalogInventoryTotals {
    programs: usize,
    nodes: Option<usize>,
    uploaded_logical_commands: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::vnext_ops::CpuVNextComposition;
    use ferrum_interfaces::vnext::{
        DeviceId, ExecutionLane, ReusableExecutionBucketSpec, ReusableExecutionCapacity,
        ReusableExecutionClassId,
    };
    use std::collections::BTreeMap;

    fn two_program_cache() -> CudaExecutableCache {
        let composition = CpuVNextComposition::create(
            DeviceId::new("device.cpu.catalog-diagnostic").unwrap(),
            1024 * 1024,
        )
        .unwrap();
        let lane = ExecutionLane::create(Arc::clone(composition.runtime())).unwrap();
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("catalog-diagnostic").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        let mut cache = CudaExecutableCache::new();
        cache
            .configure(DeviceReusableExecutionPlan::new(2).unwrap())
            .unwrap();
        for slot in 1..=2 {
            let id = DeviceReusableExecutionProgramId::new(
                serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
                "b".repeat(64),
                lane.id(),
                bucket.bucket_id().clone(),
                "c".repeat(64),
                "d".repeat(64),
                slot,
                1,
                1,
                1,
            )
            .unwrap();
            let capture = DeviceReusableExecutionCapture::new(id, 3, vec![], vec![]).unwrap();
            cache
                .register_program(
                    &capture,
                    &[],
                    &[],
                    &[],
                    &[],
                    &CudaExecutablePreparation::default(),
                )
                .unwrap();
        }
        cache
    }

    #[test]
    fn cuda_catalog_diagnostics_count_full_registry_even_when_capture_stops_at_limit() {
        let cache = two_program_cache();
        assert_eq!(
            cache.cost_catalog_inventory_totals(),
            CatalogInventoryTotals {
                programs: 2,
                nodes: Some(6),
                uploaded_logical_commands: Some(0),
            }
        );
        let limits = DeviceCostGraphCatalogLimits::new(2, 5, 5).unwrap();
        let error = cache
            .cost_reusable_graph_catalog(limits, &mut || Ok(()))
            .unwrap_err();
        let reason = error.to_string();
        let detailed = cache
            .cost_catalog_diagnostic_error(error, limits)
            .to_string();
        assert!(detailed.starts_with(&reason));
        assert!(detailed.contains("programs=2, nodes=Some(6), uploaded_logical_commands=Some(0)"));
        let cancelled = cache
            .cost_reusable_graph_catalog_shared(limits, &mut || {
                Err(ferrum_interfaces::vnext::VNextError::InvalidExecutionPlan {
                    reason: "cancelled capture".into(),
                })
            })
            .unwrap_err()
            .to_string();
        assert!(cancelled.contains("cancelled capture"));
        assert!(!cancelled.contains("CUDA cost catalog inventory"));
    }

    #[derive(Clone)]
    struct Events {
        enabled: bool,
        records: Arc<std::sync::Mutex<Vec<BTreeMap<String, String>>>>,
        exhausted: Arc<std::sync::atomic::AtomicBool>,
    }

    impl tracing::Subscriber for Events {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            self.enabled && metadata.target() == "ferrum::cost_catalog_diagnostics"
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            struct Fields(BTreeMap<String, String>);
            impl tracing::field::Visit for Fields {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    self.0.insert(field.name().to_owned(), format!("{value:?}"));
                }
            }
            let mut fields = Fields(BTreeMap::new());
            event.record(&mut fields);
            self.records.lock().unwrap().push(fields.0);
            // Deterministic expiry at the diagnostic boundary, without wall
            // clock races. The caller's final budget check remains required.
            self.exhausted
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn events(enabled: bool) -> Events {
        Events {
            enabled,
            records: Arc::new(std::sync::Mutex::new(Vec::new())),
            exhausted: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    #[test]
    fn cuda_catalog_original_failure_is_recorded_before_optional_inventory_and_budget_expiry() {
        use std::sync::atomic::Ordering;
        let cache = two_program_cache();
        let limits = DeviceCostGraphCatalogLimits::new(2, 5, 5).unwrap();
        let observer = events(true);
        let error = tracing::subscriber::with_default(observer.clone(), || {
            cache
                .cost_reusable_graph_catalog_shared(limits, &mut || {
                    if observer.exhausted.load(Ordering::Relaxed) {
                        Err(ferrum_interfaces::vnext::VNextError::InvalidExecutionPlan {
                            reason: "budget exhausted".into(),
                        })
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err()
                .to_string()
        });
        assert!(observer.exhausted.load(Ordering::Relaxed));
        assert!(error.contains("CUDA cost catalog inventory"));
        let records = observer.records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0]["error"].contains("graph catalog node limit exceeded"));
        assert!(!records[0]["error"].contains("CUDA cost catalog inventory"));
        assert_eq!(records[0]["maximum_programs"], "2");
        assert_eq!(records[0]["maximum_nodes"], "5");
        assert_eq!(records[0]["maximum_logical_commands"], "5");
        assert!(cache.cost_catalog.get().is_none());
    }

    #[test]
    fn cuda_catalog_disabled_or_cancelled_diagnostics_do_not_emit_or_append_inventory() {
        let cache = two_program_cache();
        let limits = DeviceCostGraphCatalogLimits::new(2, 5, 5).unwrap();
        let disabled = events(false);
        let error = tracing::subscriber::with_default(disabled.clone(), || {
            cache
                .cost_reusable_graph_catalog_shared(limits, &mut || Ok(()))
                .unwrap_err()
                .to_string()
        });
        assert!(error.contains("graph catalog node limit exceeded"));
        assert!(!error.contains("CUDA cost catalog inventory"));
        assert!(disabled.records.lock().unwrap().is_empty());
        let enabled = events(true);
        let error = tracing::subscriber::with_default(enabled.clone(), || {
            cache
                .cost_reusable_graph_catalog_shared(limits, &mut || {
                    Err(ferrum_interfaces::vnext::VNextError::InvalidExecutionPlan {
                        reason: "cancelled capture".into(),
                    })
                })
                .unwrap_err()
                .to_string()
        });
        assert!(error.contains("cancelled capture"));
        assert!(!error.contains("CUDA cost catalog inventory"));
        assert!(enabled.records.lock().unwrap().is_empty());
    }
}

impl CudaExecutableCache {
    fn cost_catalog_inventory_totals(&self) -> CatalogInventoryTotals {
        let mut nodes = Some(0usize);
        let mut commands = Some(0usize);
        for program in self.programs.values() {
            nodes = nodes.and_then(|n| n.checked_add(program.descriptor.node_count() as usize));
            for segment in &program.segments {
                if !self
                    .entries
                    .get(&segment.key)
                    .is_some_and(|entry| entry.uploaded)
                {
                    continue;
                }
                if let Some(logical) = &segment.logical_commands {
                    commands = commands.and_then(|n| n.checked_add(logical.len()));
                }
            }
        }
        CatalogInventoryTotals {
            programs: self.programs.len(),
            nodes,
            uploaded_logical_commands: commands,
        }
    }

    pub(super) fn cost_catalog_diagnostic_error(
        &self,
        error: CudaDeviceRuntimeError,
        limits: DeviceCostGraphCatalogLimits,
    ) -> CudaDeviceRuntimeError {
        let totals = self.cost_catalog_inventory_totals();
        CudaDeviceRuntimeError::contract(format!(
            "{error}; CUDA cost catalog inventory: programs={}, nodes={:?}, uploaded_logical_commands={:?}; limits: programs={}, nodes={}, logical_commands={}",
            totals.programs,
            totals.nodes,
            totals.uploaded_logical_commands,
            limits.maximum_programs(),
            limits.maximum_nodes(),
            limits.maximum_logical_commands(),
        ))
    }
}
