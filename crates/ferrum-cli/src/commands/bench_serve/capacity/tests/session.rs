use super::*;

#[tokio::test]
async fn capacity_session_http_reuses_only_warmup_and_preserves_measured_prompt_indices() {
    let mut contract = http_contract();
    let mut warm = contract.workload.samples[0].clone();
    warm.phase = BenchmarkPhase::Warmup;
    warm.prompt_sha256 = sha256_hex(b"warmup prompt");
    contract.workload.samples.insert(0, warm);
    let digest = sha256_hex(&serde_json::to_vec(&contract.workload.samples).unwrap());
    contract.identity.ordered_workload_sha256 = digest.clone();
    contract.workload.selection_sha256 = digest;
    // Intentionally unattainable latency distinguishes a valid FAIL from a
    // broken acquisition; both successive measured blocks must still execute.
    contract.slo.ttft_ms = 0.000001;
    let mut search = CapacitySearch::new(contract).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = observed.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let request = read_request(&mut socket).await;
                let start = request.windows(4).position(|x| x == b"\r\n\r\n").unwrap() + 4;
                let body: serde_json::Value = serde_json::from_slice(&request[start..]).unwrap();
                seen.lock()
                    .unwrap()
                    .push(body["messages"][0]["content"].as_str().unwrap().to_owned());
                let body = concat!(
                    "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
                    "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n",
                    "data: [DONE]\n\n");
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                socket.write_all(response.as_bytes()).await.unwrap();
            });
        }
    });
    let mut inputs = prompts();
    inputs.insert(
        0,
        PromptCase {
            text: "warmup prompt".into(),
            input_tokens: 2,
            sha256: sha256_hex(b"warmup prompt"),
            output_budget: Some(2),
        },
    );
    let identity = CapacityServerSession {
        session_id: "HTTP fixture process".into(),
        process_id: std::process::id(),
        process_birth: CapacityProcessBirth::UnixStartTime {
            seconds: 1,
            microseconds: 0,
        },
        server_binary_sha256: "a".repeat(64),
        effective_configuration_sha256: "a".repeat(64),
        process_configuration_sha256: "b".repeat(64),
        endpoint: format!("http://{address}"),
        engine_instance: None,
        started_unix_ns: unix_ns().unwrap(),
    };
    let mut origin: Option<String> = None;
    let mut previous = None;
    let mut measured = 0;
    for ordinal in 1..=2 {
        let SearchProgress::Awaiting { runs, .. } = search.progress() else {
            panic!("pending block");
        };
        let plan = search.planned_run(&runs[0]).unwrap();
        let block = CapacitySessionBlock {
            session: identity.clone(),
            key: plan.key.clone(),
            block_ordinal: ordinal,
            checked_live_unix_ns: unix_ns().unwrap(),
            previous_evidence_sha256: previous,
        };
        let authorization = search
            .authorize_session_block(&block, origin.as_deref())
            .unwrap();
        let mut ctx = context(identity.endpoint.clone());
        let run_id = format!("capacity-session-{ordinal}");
        ctx.benchmark_run_id = Arc::new(run_id.clone());
        let started = unix_ns().unwrap();
        let run = run_fixed_window(
            &ctx,
            &plan,
            &inputs,
            session::warmup_count(Some(&authorization), 1),
            &search.contract().window,
            search.contract().maximum_queue_samples_per_run,
        )
        .await
        .unwrap();
        let mut evidence = assemble_evidence(
            search.contract(),
            &plan,
            run,
            run_id,
            started,
            unix_ns().unwrap(),
        )
        .unwrap();
        session::bind(&mut evidence, Some(authorization));
        assert!(evidence
            .request_records
            .iter()
            .all(|r| r.workload_sample_index == 1));
        assert_eq!(evidence.warmup.completed, u32::from(ordinal == 1));
        measured += evidence.requests.len();
        let digest = capacity_evidence_sha256(&evidence).unwrap();
        let assessment = search.record(evidence).unwrap();
        assert_eq!(
            assessment.acquisition_disposition,
            CapacityAcquisitionDisposition::Complete,
            "{:?}",
            assessment.issues
        );
        assert_eq!(assessment.status, SloStatus::Fail);
        if origin.is_none() {
            origin = Some(digest.clone());
        }
        previous = Some(digest);
    }
    server.abort();
    let _ = server.await;
    let seen = observed.lock().unwrap();
    assert_eq!(seen.len(), measured + 1);
    assert_eq!(
        seen.iter()
            .filter(|v| v.as_str() == "warmup prompt")
            .count(),
        1
    );
    assert_eq!(
        seen.iter().filter(|v| v.as_str() == "test prompt").count(),
        measured
    );
}
