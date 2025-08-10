use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_project::calculator::*;

fn benchmark_add(c: &mut Criterion) {
    c.bench_function("add", |b| b.iter(|| add(black_box(100), black_box(200))));
}

fn benchmark_multiply(c: &mut Criterion) {
    c.bench_function("multiply", |b| {
        b.iter(|| multiply(black_box(100), black_box(200)))
    });
}

fn benchmark_divide(c: &mut Criterion) {
    c.bench_function("divide", |b| {
        b.iter(|| divide(black_box(1000), black_box(10)))
    });
}

fn benchmark_operations_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("calculator_operations");

    group.bench_function("add_small", |b| b.iter(|| add(black_box(1), black_box(2))));

    group.bench_function("add_large", |b| {
        b.iter(|| add(black_box(1_000_000), black_box(2_000_000)))
    });

    group.bench_function("multiply_small", |b| {
        b.iter(|| multiply(black_box(10), black_box(20)))
    });

    group.bench_function("multiply_large", |b| {
        b.iter(|| multiply(black_box(10_000), black_box(20_000)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_add,
    benchmark_multiply,
    benchmark_divide,
    benchmark_operations_group
);
criterion_main!(benches);
