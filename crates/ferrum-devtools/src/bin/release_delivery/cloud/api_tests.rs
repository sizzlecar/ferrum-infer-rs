use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

pub(crate) struct Reply {
    pub(crate) method: &'static str,
    pub(crate) path: &'static str,
    pub(crate) status: u16,
    pub(crate) body: Value,
}
async fn fixture(replies: Vec<Reply>) -> (Client, tokio::task::JoinHandle<Vec<Value>>) {
    let (base, server) = http_fixture(replies).await;
    (Client::for_test(base), server)
}
pub(crate) async fn http_fixture(
    replies: Vec<Reply>,
) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        for reply in replies {
            let (mut socket, _) = tokio::time::timeout(Duration::from_secs(15), listener.accept())
                .await
                .unwrap()
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 4096];
            let end = loop {
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&buffer[..read]);
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let header = String::from_utf8(bytes[..end].to_vec()).unwrap();
            let first = header.lines().next().unwrap();
            let fields: Vec<_> = first.split_whitespace().collect();
            assert_eq!(fields[0], reply.method);
            assert!(
                fields[1].starts_with(reply.path),
                "unexpected fixture endpoint"
            );
            let length = header
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while bytes.len() < end + length {
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&buffer[..read]);
            }
            let body = if length > 0 {
                serde_json::from_slice(&bytes[end..end + length]).unwrap()
            } else {
                Value::Null
            };
            // Do not retain Authorization headers in fixtures or evidence.
            requests.push(json!({"method":fields[0],"path":fields[1],"body":body}));
            let body = serde_json::to_vec(&reply.body).unwrap();
            let response=format!("HTTP/1.1 {} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",reply.status,body.len());
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
        }
        requests
    });
    (base, server)
}
fn row(id: u64, label: &str) -> Value {
    json!({"id":id,"label":label,"actual_status":"running","ssh_host":"ssh5.vast.ai","ssh_port":12122})
}

fn offer(id: u64) -> Offer {
    serde_json::from_value(
        json!({"id":id,"gpu_name":"RTX 6000Ada","num_gpus":1,"gpu_ram":49140,"cpu_ram":128970,
        "compute_cap":890,"cpu_arch":"amd64","driver_version":"580.95.05","disk_space":400.0,
        "dph_total":0.65,"inet_down_cost":0.003,"inet_up_cost":0.003,"duration":86400,
        "rentable":true,"rented":false,"verification":"verified","is_bid":false,"external":null}),
    )
    .unwrap()
}

fn unavailable(id: u64) -> Value {
    json!({"success":false,"error":"invalid_args",
        "msg":format!("error 404/3603: no_such_ask  Instance type by id {id} is not available."),"ask_id":id})
}

#[test]
fn cloud_unavailable_offer_requires_a_specific_matching_rejection() {
    use reqwest::StatusCode;
    assert!(unavailable_offer(
        StatusCode::BAD_REQUEST,
        &unavailable(7),
        7
    ));
    let modern = json!({"success":false,"error":"no_such_ask","ask_id":7});
    assert!(unavailable_offer(StatusCode::GONE, &modern, 7));
    assert!(unavailable_offer(StatusCode::NOT_FOUND, &modern, 7));
    for status in [
        StatusCode::OK,
        StatusCode::UNAUTHORIZED,
        StatusCode::FORBIDDEN,
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::BAD_GATEWAY,
    ] {
        assert!(!unavailable_offer(status, &modern, 7));
        assert!(!unavailable_offer(status, &unavailable(7), 7));
    }
    for value in [
        unavailable(8),
        json!({"success":true,"error":"no_such_ask","ask_id":7}),
        json!({"success":false,"error":"no_such_ask"}),
        json!({"success":false,"error":"invalid_args","msg":"invalid disk size","ask_id":7}),
        json!({"success":false,"error":"invalid_args","msg":"upstream timeout mentioning no_such_ask","ask_id":7}),
        json!({"error":"no_such_ask","ask_id":7}),
    ] {
        assert!(!unavailable_offer(StatusCode::BAD_REQUEST, &value, 7));
    }
}

#[tokio::test]
async fn cloud_next_offer_requires_rejection_and_complete_empty_ownership_lookup() {
    let (client, server) = fixture(vec![
        Reply {
            method: "PUT",
            path: "/api/v0/asks/7/",
            status: 400,
            body: unavailable(7),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[row(99,"another-task")],"next_token":"next"}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":null}),
        },
        Reply {
            method: "PUT",
            path: "/api/v0/asks/8/",
            status: 200,
            body: json!({"success":true,"new_contract":41}),
        },
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut args = super::super::tests::args();
    args.report_dir = directory.path().to_owned();
    args.max_create_attempts = 3;
    let mut owned = Vec::new();
    let mut unconfirmed = false;
    let id = super::super::acquire_from_offers(
        &client,
        &args,
        "owned",
        &mut owned,
        &mut unconfirmed,
        &[offer(7), offer(7), offer(8)],
    )
    .await
    .unwrap();
    assert_eq!(id, 41);
    assert_eq!(owned, vec![41]);
    assert!(!unconfirmed);
    let requests = server.await.unwrap();
    let creates: Vec<_> = requests
        .iter()
        .filter(|request| request["method"] == "PUT")
        .collect();
    assert_eq!(creates.len(), 2);
    for create in creates {
        assert_eq!(create["body"]["disk"], 300);
        assert_eq!(create["body"]["label"], "owned");
        assert_eq!(create["body"]["image"], super::super::IMAGE);
    }
    let attempts: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("create-attempts.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(attempts[0]["status"], "offer_unavailable");
    assert_eq!(attempts[1]["status"], "allocated");
    let current: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("create-intent.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(current["offer"]["id"], 8);
}

#[tokio::test]
async fn cloud_confirmed_rejections_respect_the_create_attempt_budget() {
    let (client, server) = fixture(vec![
        Reply {
            method: "PUT",
            path: "/api/v0/asks/7/",
            status: 410,
            body: json!({"success":false,"error":"no_such_ask","ask_id":7}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":null}),
        },
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut args = super::super::tests::args();
    args.report_dir = directory.path().to_owned();
    let mut owned = Vec::new();
    let mut unconfirmed = false;
    let error = super::super::acquire_from_offers(
        &client,
        &args,
        "owned",
        &mut owned,
        &mut unconfirmed,
        &[offer(7), offer(8)],
    )
    .await
    .unwrap_err();
    assert!(error.contains("1 confirmed rejections"));
    assert!(owned.is_empty());
    assert!(
        !unconfirmed,
        "definitively absent allocation must not be reported as an orphan"
    );
    server.await.unwrap();
}

#[tokio::test]
async fn cloud_unavailable_offer_does_not_advance_with_an_instance_or_failed_lookup() {
    for (status, body, expected_owned) in [
        (
            200,
            json!({"instances":[row(41,"owned")],"next_token":null}),
            vec![41],
        ),
        (503, json!({"error":"upstream unavailable"}), vec![]),
    ] {
        let (client, server) = fixture(vec![
            Reply {
                method: "PUT",
                path: "/api/v0/asks/7/",
                status: 400,
                body: unavailable(7),
            },
            Reply {
                method: "GET",
                path: "/api/v1/instances/",
                status,
                body,
            },
        ])
        .await;
        let directory = tempfile::tempdir().unwrap();
        let mut args = super::super::tests::args();
        args.report_dir = directory.path().to_owned();
        args.max_create_attempts = 3;
        let mut owned = Vec::new();
        let mut unconfirmed = false;
        assert!(super::super::acquire_from_offers(
            &client,
            &args,
            "owned",
            &mut owned,
            &mut unconfirmed,
            &[offer(7), offer(8)]
        )
        .await
        .is_err());
        assert_eq!(owned, expected_owned);
        assert!(unconfirmed);
        server.await.unwrap();
    }
}

#[tokio::test]
async fn cloud_uncertain_creation_never_tries_the_next_offer() {
    let mut replies = vec![Reply {
        method: "PUT",
        path: "/api/v0/asks/7/",
        status: 502,
        body: json!({"success":false,"error":"no_such_ask","ask_id":7,"private":"must not appear in errors"}),
    }];
    for _ in 0..3 {
        replies.push(Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":null}),
        });
    }
    let (client, server) = fixture(replies).await;
    let directory = tempfile::tempdir().unwrap();
    let mut args = super::super::tests::args();
    args.report_dir = directory.path().to_owned();
    args.max_create_attempts = 3;
    let mut owned = Vec::new();
    let mut unconfirmed = false;
    let error = super::super::acquire_from_offers(
        &client,
        &args,
        "owned",
        &mut owned,
        &mut unconfirmed,
        &[offer(7), offer(8)],
    )
    .await
    .unwrap_err();
    assert!(error.contains("no unique instance reconciled"));
    assert!(!error.contains("must not appear"));
    assert!(owned.is_empty());
    assert!(unconfirmed);
    let attempts: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("create-attempts.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(attempts.as_array().unwrap().len(), 1);
    server.await.unwrap();
}

#[tokio::test]
async fn cloud_cancelled_reconciliation_preserves_the_pending_create() {
    let (client, server) = fixture(vec![
        Reply {
            method: "PUT",
            path: "/api/v0/asks/7/",
            status: 502,
            body: json!({}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":null}),
        },
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut args = super::super::tests::args();
    args.report_dir = directory.path().to_owned();
    args.max_create_attempts = 3;
    let mut owned = Vec::new();
    let mut unconfirmed = false;
    let offers = [offer(7), offer(8)];
    {
        let operation = super::super::acquire_from_offers(
            &client,
            &args,
            "owned",
            &mut owned,
            &mut unconfirmed,
            &offers,
        );
        tokio::pin!(operation);
        // Drop the acquisition future after the first ownership response. This
        // tests cancellation without depending on scheduler timing or a sleep.
        tokio::select! {
            result = &mut operation => panic!("unexpected completion before reconciliation: {result:?}"),
            requests = server => { requests.unwrap(); }
        }
    }
    assert!(unconfirmed);
    assert!(owned.is_empty());
}

#[tokio::test]
async fn cloud_lost_create_response_reconciles_without_a_second_allocation() {
    let (client, server) = fixture(vec![
        Reply {
            method: "PUT",
            path: "/api/v0/asks/7/",
            status: 200,
            body: json!({"success":true}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[row(41,"owned")],"next_token":null}),
        },
    ])
    .await;
    let mut owned = Vec::new();
    let args = super::super::tests::args();
    assert_eq!(
        super::super::acquire(&client, 7, &args, "owned", &mut owned)
            .await
            .unwrap(),
        Some(41)
    );
    assert_eq!(owned, vec![41]);
    let requests = server.await.unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|request| request["method"] == "PUT")
            .count(),
        1
    );
    assert_eq!(requests[0]["body"]["disk"], 300);
    assert_eq!(requests[0]["body"]["runtype"], "ssh_proxy");
    assert_eq!(requests[0]["body"]["cancel_unavail"], true);
}

#[tokio::test]
async fn cloud_duplicate_label_is_not_an_execution_success() {
    let (client, server) = fixture(vec![
        Reply {
            method: "PUT",
            path: "/api/v0/asks/7/",
            status: 502,
            body: json!({"private":"never expose error body"}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[row(41,"owned"),row(42,"owned")],"next_token":null}),
        },
    ])
    .await;
    let mut owned = Vec::new();
    assert!(super::super::acquire(
        &client,
        7,
        &super::super::tests::args(),
        "owned",
        &mut owned
    )
    .await
    .unwrap_err()
    .contains("multiple"));
    assert_eq!(owned, vec![41, 42]);
    server.await.unwrap();
}

#[tokio::test]
async fn cloud_listing_follows_cursor_and_rejects_a_cycle() {
    let (client, server) = fixture(vec![
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[row(1,"a")],"next_token":"page2"}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[row(2,"b")],"next_token":null}),
        },
    ])
    .await;
    assert_eq!(client.instances().await.unwrap().len(), 2);
    let requests = server.await.unwrap();
    assert!(requests[1]["path"]
        .as_str()
        .unwrap()
        .contains("after_token=page2"));
    let (client, server) = fixture(vec![
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":"same"}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":"same"}),
        },
    ])
    .await;
    assert!(client.instances().await.unwrap_err().contains("repeated"));
    server.await.unwrap();
}

#[tokio::test]
async fn cloud_single_instance_requires_object_and_exact_id() {
    let (client, server) = fixture(vec![
        Reply {
            method: "GET",
            path: "/api/v0/instances/41/",
            status: 200,
            body: json!({"instances":row(41,"owned")}),
        },
        Reply {
            method: "GET",
            path: "/api/v0/instances/41/",
            status: 200,
            body: json!({"instances":[row(41,"owned")]}),
        },
        Reply {
            method: "GET",
            path: "/api/v0/instances/41/",
            status: 200,
            body: json!({"instances":row(42,"owned")}),
        },
    ])
    .await;
    assert_eq!(
        client.instance(41).await.unwrap().unwrap().ssh_port,
        Some(12122)
    );
    assert!(client.instance(41).await.is_err());
    assert!(client.instance(41).await.is_err());
    server.await.unwrap();
}

#[tokio::test]
async fn cloud_cleanup_confirms_absence_after_a_lost_delete_response() {
    let (client, server) = fixture(vec![
        Reply {
            method: "DELETE",
            path: "/api/v0/instances/41/",
            status: 502,
            body: json!({}),
        },
        Reply {
            method: "GET",
            path: "/api/v1/instances/",
            status: 200,
            body: json!({"instances":[],"next_token":null}),
        },
    ])
    .await;
    super::super::destroy_confirm(&client, 41).await.unwrap();
    server.await.unwrap();
}

#[test]
fn cloud_offer_filter_checks_returned_price_hardware_and_on_demand_status() {
    let mut value = json!({"id":7,"gpu_name":"RTX 6000Ada","num_gpus":1,"gpu_ram":49140,"cpu_ram":128970,
        "compute_cap":890,"cpu_arch":"amd64","driver_version":"580.95.05","disk_space":400.0,
        "dph_total":0.655555556,"inet_down_cost":0.00390625,"inet_up_cost":0.00390625,"duration":86400,
        "rentable":true,"rented":false,"verification":"verified","is_bid":false,"external":null});
    let input = super::super::tests::args();
    assert!(serde_json::from_value::<Offer>(value.clone())
        .unwrap()
        .eligible(&input));
    for (field, wrong) in [
        ("dph_total", json!(0.8)),
        ("inet_up_cost", json!(0.1)),
        ("gpu_name", json!("RTX 4090")),
        ("num_gpus", json!(2)),
        ("compute_cap", json!(860)),
        ("duration", json!(600)),
        ("is_bid", json!(true)),
        ("external", json!(true)),
    ] {
        let previous = value[field].clone();
        value[field] = wrong;
        assert!(
            !serde_json::from_value::<Offer>(value.clone())
                .unwrap()
                .eligible(&input),
            "{field}"
        );
        value[field] = previous;
    }
}
