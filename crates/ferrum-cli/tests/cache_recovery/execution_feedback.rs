//! The optional feedback owner is assembled in both real product entrypoints.
use super::*;

fn with_feedback(mut flow: Flow) -> Flow {
    flow.arguments.extend([
        "--scheduler-slo".to_owned(),
        "ttft:200,tpot:15,itl:50".to_owned(),
    ]);
    flow.expect_execution_feedback = true;
    flow
}

pub(super) async fn assert_worker_observed_execution(client: &reqwest::Client, url: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let health: Value = client
                .get(format!("{url}/health"))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .json()
                .await
                .unwrap();
            let slo = &health["scheduler"]["slo"];
            assert_eq!(slo["enabled"], true, "{health}");
            assert_eq!(slo["feedback"]["worker_failures"], 0, "{health}");
            if slo["feedback"]["trained_samples"].as_u64().unwrap() > 0 {
                assert!(
                    slo["feedback"]["accepted"].as_u64().unwrap() > 0,
                    "{health}"
                );
                assert!(slo["planning_rounds"].as_u64().unwrap() > 0, "{health}");
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("real execution must reach the independently owned feedback worker");
}

#[tokio::test]
async fn run_feedback_keeps_real_output_and_exits_cleanly() {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
    with_feedback(Flow::run(REPO))
        .successful(&hub, cache.path())
        .await;
}

#[cfg(unix)]
#[tokio::test]
async fn serve_feedback_coexists_with_prefix_policy_and_graceful_shutdown() {
    for prefix_cache in [false, true] {
        let cache = tempfile::tempdir().unwrap();
        let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
        let mut flow = with_feedback(Flow::serve(REPO).await);
        flow.arguments.push(
            if prefix_cache {
                "--enable-prefix-cache"
            } else {
                "--disable-prefix-cache"
            }
            .to_owned(),
        );
        flow.successful_with_shutdown(&hub, cache.path(), Shutdown::Terminate)
            .await;
    }
}
