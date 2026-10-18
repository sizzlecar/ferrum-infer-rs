use ferrum_runtime::ResolutionSourceArtifact;

fn main() {
    let _: ResolutionSourceArtifact = serde_json::from_str("{}").unwrap();
}
