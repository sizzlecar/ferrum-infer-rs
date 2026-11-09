//! Four F32 visual tensors feed the real mixed W4/W8 stack. The scalar oracle
//! includes replacement and post-layer injections, while B1 execution and a
//! second-request perturbation independently check the packed row semantics.
use super::*;
use crate::qwen3_vl_text::{Qwen3VLTextDeepStack, Qwen3VLTextVisualRun};

fn visual_config() -> Qwen3VLTextComfyConfig {
    config(
        text(5, 2)
            .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
            .unwrap(),
        false,
    )
    .with_visual_inputs(
        Qwen3VLTextDeepStack::new(vec![
            Qwen3VLTextVisualRun::new(1, 2).unwrap(),
            Qwen3VLTextVisualRun::new(4, 1).unwrap(),
        ])
        .unwrap(),
    )
    .unwrap()
}
fn visual_inputs(c: &Qwen3VLTextComfyConfig) -> BTreeMap<ProgramValueId, Vec<f32>> {
    let mut raw = inputs(c);
    let h = c.text().architecture().hidden_size() as usize;
    for (stream, port) in std::iter::once(Family::VISUAL_EMBEDDINGS_ID)
        .chain(Family::DEEPSTACK_IDS)
        .enumerate()
    {
        raw.insert(
            id(port),
            (0..3 * h)
                .map(|i| {
                    let sign = if (i + stream) % 3 == 0 { -1.0 } else { 1.0 };
                    sign * (1 + (i * 13 + stream * 7) % 29) as f32 / 257.0
                })
                .collect(),
        );
    }
    // Replaced placeholders have no mathematical effect on decoder success.
    raw.get_mut(&id::<ProgramValueId>(Family::EMBEDDINGS_ID))
        .unwrap()[h..3 * h]
        .fill(f32::from_bits(0x7f800001));
    raw
}
fn assert_oracle(
    actual: &BTreeMap<ProgramValueId, Vec<f32>>,
    expected: &BTreeMap<ProgramValueId, Vec<f32>>,
) {
    for (port, values) in expected {
        assert_eq!(actual[port].len(), values.len());
        for (&got, &want) in actual[port].iter().zip(values) {
            assert!(got.is_finite());
            assert!(
                (got - want).abs() <= 3.0e-5 * want.abs().max(1.0),
                "{port}: CUDA {got}, scalar {want}"
            );
        }
    }
}
#[test]
#[ignore = "requires configured CUDA host with admitted five-layer visual TE fixture capacity"]
fn actual_stored_visual_stack_preserves_three_injected_taps_and_request_isolation() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let execution = Execution::new();
    let c = visual_config();
    let (_directory, source) = archive(&original(&c));
    let raw = visual_inputs(&c);
    let expected = oracle::execute(&c, &source, &raw);
    let actual = execution.execute(&c, &source, &raw, "visual.b2");
    assert_oracle(&actual, &expected);
    let h = c.text().architecture().hidden_size() as usize;
    let s = c.text().layout().sequence_length() as usize;
    for sample in 0..2 {
        let one_text = Qwen3VLTextConfig::new(
            c.text().architecture().clone(),
            Qwen3VLTextLayout::new(s as u32, vec![c.text().layout().key_runs()[sample].clone()])
                .unwrap(),
        )
        .unwrap()
        .with_hidden_state_indices(c.text().hidden_state_indices().to_vec())
        .unwrap();
        let one = Qwen3VLTextComfyConfig::from_checkpoint(one_text, &source)
            .unwrap()
            .with_visual_inputs(
                Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(
                    1,
                    if sample == 0 { 2 } else { 1 },
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap();
        let one_inputs = raw
            .iter()
            .map(|(port, data)| {
                let visual = port.as_str() == Family::VISUAL_EMBEDDINGS_ID
                    || Family::DEEPSTACK_IDS.contains(&port.as_str());
                let (start, count, width) = if visual {
                    (
                        if sample == 0 { 0 } else { 2 },
                        if sample == 0 { 2 } else { 1 },
                        h,
                    )
                } else {
                    (
                        sample * s,
                        s,
                        if port.as_str() == Family::EMBEDDINGS_ID {
                            h
                        } else {
                            c.text().architecture().head_dim() as usize
                        },
                    )
                };
                (
                    port.clone(),
                    data[start * width..(start + count) * width].to_vec(),
                )
            })
            .collect();
        let one_actual =
            execution.execute(&one, &source, &one_inputs, &format!("visual.b1.{sample}"));
        for (port, values) in one_actual {
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual[&port][sample * s * h..(sample + 1) * s * h]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{port}: visual B1/B2 membership changed sample {sample}"
            );
        }
    }
    let mut changed = raw.clone();
    for port in std::iter::once(Family::VISUAL_EMBEDDINGS_ID).chain(Family::DEEPSTACK_IDS) {
        for x in &mut changed.get_mut(&id::<ProgramValueId>(port)).unwrap()[2 * h..] {
            *x = -*x * 2.125;
        }
    }
    let altered = execution.execute(&c, &source, &changed, "visual.b2.perturbed");
    assert_oracle(&altered, &oracle::execute(&c, &source, &changed));
    for (port, data) in &actual {
        assert_eq!(
            &data[..s * h],
            &altered[port][..s * h],
            "{port}: request1 visual data leaked into request0"
        );
    }
    assert!(
        actual
            .iter()
            .any(|(port, data)| data[s * h..] != altered[port][s * h..]),
        "all visual branches must affect the second request"
    );
}
