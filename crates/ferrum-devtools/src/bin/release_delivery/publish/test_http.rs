//! Small loopback protocol fixture; never contacts a registry or GitHub.
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

pub(super) struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}
pub(super) struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}
impl Response {
    pub fn json(status: u16, body: serde_json::Value) -> Self {
        Self {
            status,
            body: serde_json::to_vec(&body).unwrap(),
        }
    }
    pub fn text(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            body: body.into(),
        }
    }
}
pub(super) struct Server {
    pub url: String,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Server {
    pub fn new(mut handler: impl FnMut(Request) -> Response + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let finished = stop.clone();
        let worker = thread::spawn(move || {
            while !finished.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // Accepted sockets inherit O_NONBLOCK on macOS. The
                        // worker reads one complete request with a bounded timeout.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(3)))
                            .unwrap();
                        let response = handler(read_request(&mut stream));
                        write!(stream, "HTTP/1.1 {} Fixture\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n", response.status, response.body.len()).unwrap();
                        stream.write_all(&response.body).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("loopback accept: {error}"),
                }
            }
        });
        Self {
            url,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            if !thread::panicking() {
                worker.join().unwrap();
            }
        }
    }
}
fn read_request(stream: &mut TcpStream) -> Request {
    let mut data = Vec::new();
    let mut chunk = [0; 4096];
    let boundary = loop {
        if let Some(position) = data.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
        let read = stream.read(&mut chunk).unwrap();
        assert!(read > 0, "incomplete request headers");
        data.extend_from_slice(&chunk[..read]);
    };
    let headers = std::str::from_utf8(&data[..boundary]).unwrap();
    let mut first = headers.lines().next().unwrap().split_whitespace();
    let method = first.next().unwrap().to_string();
    let path = first.next().unwrap().to_string();
    let length: usize = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse().unwrap())
        .unwrap_or(0);
    while data.len() < boundary + length {
        let read = stream.read(&mut chunk).unwrap();
        assert!(read > 0, "incomplete request body");
        data.extend_from_slice(&chunk[..read]);
    }
    Request {
        method,
        path,
        body: data[boundary..boundary + length].to_vec(),
    }
}
