use ferrum_types::*;

#[test]
fn memory_usage_utilization() {
    let mut m = MemoryUsage {
        total_bytes: 100,
        used_bytes: 40,
        free_bytes: 60,
        gpu_memory_bytes: Some(0),
        cpu_memory_bytes: Some(0),
        cache_memory_bytes: 0,
        utilization_percent: 0.0,
    };
    m.calculate_utilization();
    assert!((m.utilization_percent - 40.0).abs() < 1e-3);
}
