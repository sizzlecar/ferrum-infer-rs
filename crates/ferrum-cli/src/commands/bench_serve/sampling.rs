use clap::Args;
use ferrum_bench_core::env::HttpRequestSampling;

/// Explicit generation controls shared by every HTTP benchmark workload.
#[derive(Args, Clone, Copy, Debug, Default)]
pub struct BenchSamplingArgs {
    /// Sampling temperature. Zero preserves greedy generation.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Positive top-k limit. Omitted from requests unless set.
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Nucleus probability in (0, 1]. Omitted from requests unless set.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Generation seed sent unchanged in every request; separate from --seed.
    #[arg(long)]
    pub sampling_seed: Option<u64>,
}

impl BenchSamplingArgs {
    pub(super) fn request_sampling(self) -> HttpRequestSampling {
        HttpRequestSampling {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.sampling_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{build_env, chat_completion_body, validate_command, BenchServeCommand};
    use super::*;
    use clap::Parser;
    use serde_json::{json, Value};

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        command: BenchServeCommand,
    }

    fn parse(extra: &[&str]) -> BenchServeCommand {
        TestCli::try_parse_from(
            [
                "bench",
                "--base-url",
                "http://127.0.0.1:9",
                "--model",
                "test",
                "--tokenizer",
                ".",
            ]
            .into_iter()
            .chain(extra.iter().copied()),
        )
        .unwrap()
        .command
    }

    fn body(command: &BenchServeCommand) -> Value {
        chat_completion_body(
            "test",
            "prompt",
            16,
            false,
            None,
            None,
            command.sampling.request_sampling(),
        )
    }

    #[test]
    fn default_sampling_preserves_greedy_wire_and_prompt_seed_is_not_sent() {
        let command = parse(&["--seed", "11"]);
        validate_command(&command).unwrap();
        assert_eq!(command.seed, Some(11));
        assert_eq!(
            body(&command),
            json!({
                "model": "test",
                "messages": [{"role": "user", "content": "prompt"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": true,
                "stream_options": {"include_usage": true}
            })
        );
        assert_eq!(
            build_env(&command, vec![]).http_request_sampling,
            Some(HttpRequestSampling::default())
        );
    }

    #[test]
    fn cli_sampling_matches_wire_and_report_without_replacing_prompt_seed() {
        let command = parse(&[
            "--seed",
            "11",
            "--sampling-seed",
            "37",
            "--temperature",
            "0.6",
            "--top-k",
            "20",
            "--top-p",
            "0.95",
        ]);
        validate_command(&command).unwrap();
        assert_eq!(command.seed, Some(11));
        let expected = HttpRequestSampling {
            temperature: 0.6,
            top_k: Some(20),
            top_p: Some(0.95),
            seed: Some(37),
        };
        let wire = body(&command);
        for (name, value) in serde_json::to_value(expected).unwrap().as_object().unwrap() {
            assert_eq!(wire.get(name), Some(value), "{name}");
        }
        assert_eq!(
            build_env(&command, vec![]).http_request_sampling,
            Some(expected)
        );
        let changed_prompt_seed = parse(&[
            "--seed",
            "12",
            "--sampling-seed",
            "37",
            "--temperature",
            "0.6",
            "--top-k",
            "20",
            "--top-p",
            "0.95",
        ]);
        assert_eq!(body(&changed_prompt_seed), wire);
    }

    #[test]
    fn sampling_rejects_invalid_and_non_finite_cli_values_before_requests() {
        for argument in [
            "--temperature=-1",
            "--temperature=NaN",
            "--temperature=inf",
            "--temperature=-inf",
            "--top-p=0",
            "--top-p=-0.1",
            "--top-p=1.1",
            "--top-p=NaN",
            "--top-p=inf",
            "--top-p=-inf",
            "--top-k=0",
        ] {
            let command = parse(&[argument]);
            assert!(validate_command(&command).is_err(), "accepted {argument}");
        }
        let boundary = parse(&[
            "--temperature",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1",
            "--sampling-seed",
            "0",
        ]);
        validate_command(&boundary).unwrap();
        assert_eq!(body(&boundary)["seed"], 0);
        assert_eq!(body(&boundary)["top_k"], 1);
        assert_eq!(body(&boundary)["top_p"], 1.0);
    }
}
