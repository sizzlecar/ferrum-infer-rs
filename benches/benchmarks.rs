use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferrum_infer::config::Config;
use ferrum_infer::utils::*;

fn benchmark_validate_model_name(c: &mut Criterion) {
    c.bench_function("validate_model_name", |b| {
        b.iter(|| validate_model_name(black_box("gpt-3.5-turbo")))
    });
}

fn benchmark_sanitize_for_logging(c: &mut Criterion) {
    let text = "This is a test string with\ninvalid\0characters\tand more.";
    c.bench_function("sanitize_for_logging", |b| {
        b.iter(|| sanitize_for_logging(black_box(text), black_box(100)))
    });
}

fn benchmark_sanitize_for_logging_truncate(c: &mut Criterion) {
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100);
    c.bench_function("sanitize_for_logging_truncate", |b| {
        b.iter(|| sanitize_for_logging(black_box(&text), black_box(100)))
    });
}

fn benchmark_config_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_operations");

    group.bench_function("config_default", |b| b.iter(|| Config::default()));

    group.finish();
}

criterion_group!(
    benches,
    benchmark_validate_model_name,
    benchmark_sanitize_for_logging,
    benchmark_sanitize_for_logging_truncate,
    benchmark_config_operations
);
criterion_main!(benches);
