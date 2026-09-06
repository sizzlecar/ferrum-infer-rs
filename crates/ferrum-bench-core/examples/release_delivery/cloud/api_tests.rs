use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

struct Reply {
    method: &'static str,
    path: &'static str,
    status: u16,
    body: Value,
}
async fn fixture(replies: Vec<Reply>) -> (Client, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = Client::new("local-fixture-token".into()).unwrap();
    client.base = format!("http://{}", listener.local_addr().unwrap());
    client.http = HttpClient::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        for reply in replies {
            let (mut socket, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
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
    (client, server)
}
fn row(id: u64, label: &str) -> Value {
    json!({"id":id,"label":label,"actual_status":"running","ssh_host":"ssh5.vast.ai","ssh_port":12122})
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
        41
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
