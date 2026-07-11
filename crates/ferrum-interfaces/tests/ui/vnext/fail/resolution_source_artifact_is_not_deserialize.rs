use ferrum_interfaces::vnext::ResolutionSourceArtifact;

fn main() {
    let _: ResolutionSourceArtifact = serde_json::from_str("{}").unwrap();
}
