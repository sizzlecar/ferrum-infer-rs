use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ferrum_kernels::backend::{cpu::CpuBackend, Backend};
use ferrum_quantization::{
    NativeSafetensorsLoadOptions, NativeSafetensorsLoader, QuantConfig, QuantMethod, WeightLoader,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

struct CapturedEvents {
    messages: Arc<Mutex<Vec<String>>>,
    next_span: AtomicU64,
}

impl Subscriber for CapturedEvents {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(self.next_span.fetch_add(1, Ordering::Relaxed))
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        struct Messages<'a>(&'a mut Vec<String>);

        impl tracing::field::Visit for Messages<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push(format!("{value:?}"));
                }
            }
        }

        event.record(&mut Messages(&mut self.messages.lock().unwrap()));
    }
}

fn gptq_fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    // Eight input and output channels, one quantization group. With zero
    // code 7 and scale 1, packed codes 9 and 10 represent weights 1 and 2.
    let packed: Vec<Vec<u8>> = [0x9999_9999u32, 0xAAAA_AAAAu32]
        .iter()
        .map(|word| (0..8).flat_map(|_| word.to_le_bytes()).collect())
        .collect();
    let zeros = 0x7777_7777u32.to_le_bytes();
    let scales: Vec<_> = (0..8).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let groups: Vec<_> = (0..8).flat_map(|_| 0i32.to_le_bytes()).collect();
    let mut tensors = Vec::new();
    for (name, weights) in ["gate_proj", "up_proj"].into_iter().zip(&packed) {
        for (suffix, dtype, shape, bytes) in [
            ("qweight", Dtype::I32, vec![1, 8], weights.as_slice()),
            ("qzeros", Dtype::I32, vec![1, 1], zeros.as_slice()),
            ("scales", Dtype::F32, vec![1, 8], scales.as_slice()),
            ("g_idx", Dtype::I32, vec![8], groups.as_slice()),
        ] {
            tensors.push((
                format!("{name}.{suffix}"),
                TensorView::new(dtype, shape, bytes).unwrap(),
            ));
        }
    }
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    let config = QuantConfig {
        method: QuantMethod::Gptq,
        bits: 4,
        group_size: 8,
        desc_act: false,
        sym: true,
    };
    std::fs::write(
        directory.path().join("quantize_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    directory
}

fn assert_loader_diagnostics(loader: &NativeSafetensorsLoader<CpuBackend>, enabled: bool) {
    for (name, expected) in [
        ("gate_proj", vec![8.0; 8]),
        ("gate_up_proj", [vec![8.0; 8], vec![16.0; 8]].concat()),
    ] {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let subscriber = CapturedEvents {
            messages: messages.clone(),
            next_span: AtomicU64::new(1),
        };
        tracing::subscriber::with_default(subscriber, || {
            let linear = loader.load_linear(name).unwrap();
            let mut context = CpuBackend::new_context();
            let mut output = vec![f32::NAN; expected.len()];
            linear.forward(&mut context, &vec![1.0; 8], &mut output, 1);
            assert_eq!(output, expected);
        });
        let messages = messages.lock().unwrap();
        for diagnostic in ["GPTQ g_idx trace:", "GPTQ qzeros trace:"] {
            assert_eq!(
                messages.iter().any(|message| message.contains(diagnostic)),
                enabled,
                "{name}: diagnostic {diagnostic}, enabled={enabled}"
            );
        }
    }
}

#[test]
fn gptq_diagnostics_are_isolated_between_loader_instances() {
    let fixture = gptq_fixture();
    let enabled = NativeSafetensorsLoader::<CpuBackend>::open_with_options(
        fixture.path(),
        NativeSafetensorsLoadOptions {
            gptq_gidx_trace: true,
        },
    )
    .unwrap();
    let disabled = NativeSafetensorsLoader::<CpuBackend>::open_with_options(
        fixture.path(),
        NativeSafetensorsLoadOptions {
            gptq_gidx_trace: false,
        },
    )
    .unwrap();
    let default = NativeSafetensorsLoader::<CpuBackend>::open(fixture.path()).unwrap();

    // Both configured instances remain alive and load concurrently after the
    // default instance is constructed. Neither can change its peers' behavior.
    std::thread::scope(|scope| {
        scope.spawn(|| assert_loader_diagnostics(&enabled, true));
        scope.spawn(|| assert_loader_diagnostics(&disabled, false));
        assert_loader_diagnostics(&default, false);
    });
}
