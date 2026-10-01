//! Real original records, two concurrently retained owners, three independent
//! phases, and replay under exactly the same finite memory declaration.
use super::*;

const COMMON: [&str; 16] = [
    "memory.a", "memory.b", "memory.c", "memory.d", "memory.e", "memory.f", "memory.g", "memory.h",
    "memory.i", "memory.j", "memory.k", "memory.l", "memory.m", "memory.n", "memory.o", "memory.p",
];

struct MemoryPopulation<'a> {
    common: &'a [&'a str],
    block_offered: usize,
    phase_blocks: usize,
    max_axes: usize,
    max_rank: usize,
    source_bytes: usize,
}

impl MemoryPopulation<'_> {
    fn phase_offers(&self) -> usize {
        self.block_offered * self.phase_blocks
    }

    fn phase_members(&self) -> usize {
        // Both original owners are present in every block, in alternating order.
        assert_eq!(self.block_offered % 4, 0);
        self.phase_offers() / 2
    }

    fn load_limits(&self) -> CostProfileLoadLimits {
        CostProfileLoadLimits {
            max_file_bytes: std::num::NonZeroUsize::new(self.source_bytes).unwrap(),
            ..Default::default()
        }
    }
}

fn retained_header(
    limit: usize,
    identified: bool,
    population: &MemoryPopulation<'_>,
) -> StructuredServiceHeaderV7 {
    let mut h = header();
    if identified {
        h.declaration
            .nonnegative_envelope
            .as_mut()
            .unwrap()
            .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2;
    }
    h.declaration.schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        population.block_offered,
        [population.phase_offers(); 3],
        [population.phase_members(); 3],
        OwnerInputReadinessV1::new_cached_residual_v2([population.phase_blocks; 3], 32_000_000)
            .unwrap(),
    )
    .unwrap();
    h.declaration.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1);
    h.declaration.settings.max_phase_samples = population.phase_offers();
    h.declaration.settings.max_axes = population.max_axes;
    h.declaration.settings.max_rank = population.max_rank;
    h.declaration.maximum_retained_numeric_bytes = limit;
    h.maximum_file_bytes = population.source_bytes as u64;
    rebuild(h).unwrap()
}

fn collect(
    limit: usize,
    identified: bool,
    population: &MemoryPopulation<'_>,
) -> Result<(StructuredServiceCollectorV7, Vec<u8>), CostProfileError> {
    let h = retained_header(limit, identified, population);
    let mut bytes = record_bytes_v7(&h)?;
    let mut c = StructuredServiceCollectorV7::new(h, population.load_limits())?;
    let block_offered = population.block_offered as u64;
    for block in 1..=(1 + 3 * population.phase_blocks) as u64 {
        let first = (block - 1) * block_offered + 1;
        append(
            &mut bytes,
            &c.open_block(first * 2_000 - 1, (first - 1) * 3)?,
        );
        for ticket in first..first + block_offered {
            // Prefill remains an exact population; ordinary Decode shares the
            // frozen checked universe. No owner is paused or discarded.
            let generated = u64::from(ticket % 2 != 0);
            let mut roster = population.common.to_vec();
            roster.push(if generated == 0 || ticket % 4 == 1 {
                A
            } else {
                B
            });
            let r = record(ticket, &roster, generated);
            c.push(&r)?;
            append(&mut bytes, &r);
        }
        let r = c.close_block(paired((first + block_offered - 1) * 2_000 + 1_101))?;
        append(&mut bytes, &r);
        let audit = c.audit();
        assert_eq!(audit.owners.len(), 2, "{audit:?}");
        assert!(
            audit.owners.iter().all(|o| o.failure.is_none()),
            "{audit:?}"
        );
        if block == population.phase_blocks as u64 {
            assert!(audit.owners.iter().all(|o| o.eligible
                == (population.phase_blocks - 1) * population.block_offered / 2
                && !o.qualified));
        }
    }
    assert_eq!(c.qualified_children(), 2);
    Ok((c, bytes))
}

#[test]
fn source7_memory_accounting_retains_both_long_phases_and_replays_at_exact_capacity() {
    let population = MemoryPopulation {
        common: &COMMON,
        block_offered: 16,
        phase_blocks: 8,
        max_axes: 512,
        max_rank: 16,
        source_bytes: 16 * 1024 * 1024,
    };
    for identified in [false, true] {
        verify_retained_limits(identified, &population, None);
    }
}

#[test]
fn source7_memory_accounting_identified_wide_long_phases_replay_at_exact_capacity() {
    // A real selected-command roster creates over two hundred physical axes;
    // neither the input vectors nor the checked source observations are padded.
    // 512 members per owner/phase exercises the next sample-Vec capacity class
    // beyond a four-hundred-member phase. Both owners remain live throughout.
    let names: Vec<String> = (0..52).map(|i| format!("memory.wide.{i}")).collect();
    let common: Vec<&str> = names.iter().map(String::as_str).collect();
    let native = crate::implementations::continuous::cost_model::structured_v2::StructuredSettingsV2::default();
    let population = MemoryPopulation {
        common: &common,
        block_offered: 128,
        phase_blocks: 8,
        max_axes: native.max_axes,
        max_rank: native.max_rank,
        // Source serialization has its own product-sized file limit. The
        // numerical collector below still has exactly 64 MiB, including U.
        source_bytes: 256 * 1024 * 1024,
    };
    let mut roster = common.clone();
    roster.push(A);
    let q = query(&roster);
    let minimum_axes = population.common.len() * 4;
    assert!(q.input().regression_axes().len() >= minimum_axes);
    verify_retained_limits(true, &population, Some(minimum_axes));
}

fn verify_retained_limits(
    identified: bool,
    population: &MemoryPopulation<'_>,
    minimum_axes: Option<usize>,
) {
    let (probe, _) = collect(64 * 1024 * 1024, identified, population).unwrap();
    let required = probe.reserved_peak_for_tests();
    assert!(required < 64 * 1024 * 1024);
    drop(probe);
    // Limits affect resource admission, never sample removal or shortening.
    let (mut c, mut bytes) = collect(required, identified, population).unwrap();
    assert_eq!(
        c.offered(),
        ((1 + 3 * population.phase_blocks) * population.block_offered) as u64
    );
    assert_eq!(c.reserved_peak_for_tests(), required);
    // A checkpoint attests the original BlockClose clock, not a later read.
    let closing = paired(c.offered() * 2_000 + 1_101);
    let (r, checkpoint) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &r);
    let now = paired(c.offered() * 2_000 + 3_000);
    let limits = population.load_limits();
    let live = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(now, &limits)
        .unwrap();
    assert_eq!(live.source_sha256, replay.source_sha256);
    assert_eq!(live.children.len(), 2);
    for child in &live.children {
        assert_eq!(
            child.provenance().phases.each_ref().map(|p| p.members),
            [population.phase_members(); 3]
        );
        if let Some(minimum_axes) = minimum_axes {
            let certificate = child.model.nonnegative_fit_certificate().unwrap();
            assert!(certificate.column_maxima.len() >= minimum_axes);
            assert!(certificate.signed_basis.is_some());
            eprintln!(
                "source7 identified memory fixture: phase_members={} axes={} fit_rank={} rank_limit={} reserved_peak_bytes={required} numeric_limit_bytes={} source_bytes={}",
                population.phase_members(),
                certificate.column_maxima.len(),
                certificate.geometry_rank,
                population.max_rank,
                64 * 1024 * 1024,
                bytes.len(),
            );
        }
        let other = replay
            .children
            .iter()
            .find(|v| v.parameters_signature() == child.parameters_signature())
            .unwrap();
        assert_eq!(child.provenance().phases, other.provenance().phases);
    }
    drop((live, replay, bytes, c));
    assert!(matches!(
        collect(required - 1, identified, population),
        Err(CostProfileError::Limit(_))
    ));
}

#[test]
fn source7_memory_accounting_shared_universe_requires_the_same_arc_allocation() {
    let q = query(&[A, B]);
    let raw = q.input();
    let mut builder = crate::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseBuilderV1::new(4096, 1024 * 1024).unwrap();
    builder.observe(raw).unwrap();
    let u = builder.finish().unwrap();
    let input = raw.clone().with_algorithm_universe(&u).unwrap();
    let independent = serde_json::from_value(serde_json::to_value(&u).unwrap()).unwrap();
    assert_eq!(u, independent);
    assert_eq!(
        input.retained_bytes_with_shared_universe(Some(&independent)),
        input.retained_payload_bytes()
    );
    assert_eq!(
        input.retained_bytes_with_shared_universe(Some(&u)).unwrap()
            + u.retained_payload_bytes().unwrap(),
        input.retained_payload_bytes().unwrap()
    );
}
