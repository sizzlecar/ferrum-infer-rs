//! Loopback HTTP fixture serving exact candidate scripts or release assets.
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

pub struct Server {
    pub url: String,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
    requests: Arc<Mutex<Vec<String>>>,
}

pub struct Response {
    pub body: Vec<u8>,
    pub declared_length: Option<usize>,
}
impl From<Vec<u8>> for Response {
    fn from(body: Vec<u8>) -> Self {
        Self {
            declared_length: Some(body.len()),
            body,
        }
    }
}

impl Server {
    pub fn new(assets: BTreeMap<String, Vec<u8>>) -> Self {
        Self::responses(
            assets
                .into_iter()
                .map(|(path, body)| (path, body.into()))
                .collect(),
        )
    }

    pub fn responses(assets: BTreeMap<String, Response>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = requests.clone();
        let worker = thread::spawn(move || {
            while !signal.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // Accepted sockets can inherit the listener's nonblocking mode.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let mut request = Vec::new();
                        while !request.ends_with(b"\r\n\r\n") {
                            let mut buffer = [0; 1024];
                            let n = stream.read(&mut buffer).unwrap();
                            assert!(n > 0 && request.len() + n <= 16384, "invalid HTTP request");
                            request.extend_from_slice(&buffer[..n]);
                        }
                        let text = String::from_utf8_lossy(&request);
                        let path = text.split_whitespace().nth(1).unwrap();
                        observed.lock().unwrap().push(path.to_owned());
                        let (status, body, length) = assets
                            .get(path)
                            .map(|response| {
                                ("200 OK", response.body.as_slice(), response.declared_length)
                            })
                            .unwrap_or(("404 Not Found", b"not found", Some(9)));
                        let length = length
                            .map(|n| format!("Content-Length: {n}\r\n"))
                            .unwrap_or_default();
                        let head = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n{length}Connection: close\r\n\r\n"
                        );
                        stream.write_all(head.as_bytes()).unwrap();
                        stream.write_all(body).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => panic!("{error}"),
                }
            }
        });
        Self {
            url,
            stop,
            thread: Some(worker),
            requests,
        }
    }

    pub fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.thread.take() {
            if let Err(failure) = worker.join() {
                // Preserve the first failure during unwinding, but never turn a
                // failed server thread into a successful test on normal exit.
                if !thread::panicking() {
                    std::panic::resume_unwind(failure);
                }
            }
        }
    }
}

#[test]
fn http_server_waits_for_fragmented_request_and_returns_complete_body() {
    let server = Server::new(BTreeMap::from([("/asset".into(), b"fixture".to_vec())]));
    let mut client =
        std::net::TcpStream::connect(server.url.trim_start_matches("http://")).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    client
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    client.write_all(b"GET /asset HTTP/1.1\r\n").unwrap();
    // The server must wait for the rest of the headers, even after accept/read
    // has consumed the currently available bytes.
    thread::sleep(Duration::from_millis(30));
    client.write_all(b"Host: localhost\r\n\r\n").unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert_eq!(
        response,
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 7\r\nConnection: close\r\n\r\nfixture"
    );
    assert_eq!(server.requests(), ["/asset"]);
}

#[test]
fn http_server_failure_propagates_without_panicking_again_during_unwind() {
    fn failed_server() -> Server {
        Server {
            url: String::new(),
            stop: Arc::new(AtomicBool::new(false)),
            thread: Some(thread::spawn(|| panic!("server failure"))),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }
    let failure = std::panic::catch_unwind(|| drop(failed_server())).unwrap_err();
    assert_eq!(failure.downcast_ref::<&str>(), Some(&"server failure"));

    let failure = std::panic::catch_unwind(|| {
        let _server = failed_server();
        panic!("original test failure");
    })
    .unwrap_err();
    assert_eq!(
        failure.downcast_ref::<&str>(),
        Some(&"original test failure")
    );
}
