use super::*;

fn request() -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"image","prompt":"x"}"#))
        .unwrap()
}

#[tokio::test]
async fn single_output_hard_limit_rejects_before_inference_but_encoded_pressure_does_not() {
    for encoded_pressure in [false, true] {
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                4 * 1024 * 1024,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4096,
                    output_bytes: if encoded_pressure {
                        4 * 1024 * 1024
                    } else {
                        32
                    },
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        );
        let (mut images, calls) = image_service_with_host(host.clone(), None);
        let mut encoded_owner = None;
        if encoded_pressure {
            images.output = OutputBudget::new(OutputConfig {
                max_sessions: 2,
                encoded_bytes_total: 2 * 1024 * 1024,
                max_event_bytes: 2 * 1024 * 1024,
                ..config().output
            })
            .unwrap();
            let session = images.output.try_open_session().unwrap();
            let lease = session
                .reserve_encoded(images.output.config().encoded_bytes_total)
                .await
                .unwrap();
            encoded_owner = Some((session, lease));
        }
        let router = super::super::super::AxumServer::from_state(
            AppState::default().with_images(images.clone()),
        )
        .build_router()
        .unwrap();
        if encoded_pressure {
            // The intentionally failing generator still executes immediately;
            // only successful outputs wait for transport credit after inference.
            let response =
                tokio::time::timeout(std::time::Duration::from_secs(5), router.oneshot(request()))
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(
                images.output_usage().encoded_bytes,
                images.output.config().encoded_bytes_total
            );
            assert_eq!(
                response.status(),
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        } else {
            let response = router.oneshot(request()).await.unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
        assert_eq!(images.service().snapshot().pending_requests, 0);
        assert_eq!(images.service().snapshot().rejected_requests, 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        drop(encoded_owner);
        assert_eq!(images.output_usage().sessions, 0);
        assert_eq!(images.output_usage().encoded_bytes, 0);
        assert_eq!(images.shutdown().await.is_err(), encoded_pressure);
        assert_eq!(
            images.take_shutdown().await.unwrap().has_failure(),
            encoded_pressure
        );
    }
}

#[tokio::test]
async fn cancelling_handler_retains_active_grant_until_generation_is_quiescent() {
    let gate = Arc::new(GenerationGate::default());
    let (images, calls) = image_service_with_gate(Some(gate.clone()));
    struct ReleaseOnDrop(Arc<GenerationGate>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            *self.0.released.lock().unwrap() = true;
            self.0.changed.notify_all();
        }
    }
    let release = ReleaseOnDrop(gate.clone());
    let host = images.service().host_account().clone();
    let router = super::super::super::AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    )
    .build_router()
    .unwrap();
    let handler = tokio::spawn(router.oneshot(request()));
    gate.started.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(images.output_usage().encoded_bytes, 0);
    assert!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output)
            > 0
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        0
    );
    handler.abort();
    assert!(handler.await.unwrap_err().is_cancelled());
    assert_eq!(images.output_usage().sessions, 1);
    assert_eq!(images.output_usage().encoded_bytes, 0);
    assert!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output)
            > 0
    );
    drop(release);
    assert!(images.shutdown().await.is_err());
    drop(images.take_shutdown().await.unwrap());
    assert_eq!(images.output_usage().sessions, 0);
    assert_eq!(images.output_usage().encoded_bytes, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
