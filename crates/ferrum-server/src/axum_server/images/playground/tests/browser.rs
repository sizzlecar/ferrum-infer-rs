//! Real browser/authentication qualification; no browser download or fetch mock.
use super::*;
use anyhow::{bail, ensure, Context, Result};
use axum::{extract::Request, middleware::Next};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};

const KEY: &str = "browser-fixture-key";
const WRONG_KEY: &str = "wrong-browser-fixture-key";
const WAIT: Duration = Duration::from_secs(20);

struct Profile(PathBuf);
impl Drop for Profile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Browser {
    child: Child,
    profile: Profile,
}
impl Browser {
    fn launch() -> Result<Self> {
        let executable = PathBuf::from(
            std::env::var_os("FERRUM_TEST_BROWSER")
                .context("set FERRUM_TEST_BROWSER to an installed Chrome/Chromium executable")?,
        );
        ensure!(
            executable.is_absolute() && executable.is_file(),
            "FERRUM_TEST_BROWSER must be an absolute executable path"
        );
        let profile = Profile(std::env::temp_dir().join(format!(
            "ferrum-playground-browser-{}",
            uuid::Uuid::new_v4()
        )));
        fs::create_dir(&profile.0)?;
        let log = fs::File::create(profile.0.join("browser.log"))?;
        let child = Command::new(executable)
            .args([
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--disable-breakpad",
                "--password-store=basic",
                "--use-mock-keychain",
                "--remote-debugging-address=127.0.0.1",
                "--remote-debugging-port=0",
            ])
            .arg(format!("--user-data-dir={}", profile.0.display()))
            .arg("about:blank")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()?;
        Ok(Self { child, profile })
    }
    async fn connect(&mut self) -> Result<Cdp> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()?;
        let deadline = tokio::time::Instant::now() + WAIT;
        loop {
            if let Some(status) = self.child.try_wait()? {
                bail!("Chrome exited before CDP became available: {status}");
            }
            if let Ok(data) = fs::read_to_string(self.profile.0.join("DevToolsActivePort")) {
                if let Some(port) = data
                    .lines()
                    .next()
                    .and_then(|line| line.parse::<u16>().ok())
                {
                    if let Ok(response) = client
                        .get(format!("http://127.0.0.1:{port}/json/list"))
                        .send()
                        .await
                    {
                        if let Ok(targets) = response.json::<Vec<Value>>().await {
                            if let Some(url) = targets
                                .iter()
                                .find(|target| target["type"] == "page")
                                .and_then(|target| target["webSocketDebuggerUrl"].as_str())
                            {
                                let (socket, _) = connect_async(url).await?;
                                return Ok(Cdp { socket, next_id: 0 });
                            }
                        }
                    }
                }
            }
            ensure!(
                tokio::time::Instant::now() < deadline,
                "Chrome did not expose a page CDP endpoint"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}
impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Profile is removed only after the child has been reaped.
    }
}

struct Cdp {
    socket: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    next_id: u64,
}
impl Cdp {
    async fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.socket
            .send(Message::Text(
                json!({"id":id,"method":method,"params":params}).to_string(),
            ))
            .await?;
        tokio::time::timeout(WAIT, async {
            while let Some(message) = self.socket.next().await {
                let message = message?;
                if let Message::Text(text) = message {
                    let message: Value = serde_json::from_str(&text)?;
                    if message["id"] == id {
                        ensure!(
                            message.get("error").is_none(),
                            "CDP {method} failed: {}",
                            message["error"]
                        );
                        return Ok(message["result"].clone());
                    }
                }
            }
            bail!("CDP closed while waiting for {method}")
        })
        .await
        .context("CDP command timed out")?
    }
    async fn evaluate(&mut self, expression: &str) -> Result<Value> {
        let result = self
            .call(
                "Runtime.evaluate",
                json!({"expression":expression,"awaitPromise":true,"returnByValue":true}),
            )
            .await?;
        ensure!(
            result.get("exceptionDetails").is_none(),
            "browser expression failed: {}",
            result["exceptionDetails"]
        );
        Ok(result["result"]["value"].clone())
    }
    async fn wait_for(&mut self, condition: &str) -> Result<()> {
        let deadline = tokio::time::Instant::now() + WAIT;
        loop {
            if self
                .evaluate(&format!("Boolean({condition})"))
                .await
                .is_ok_and(|value| value == true)
            {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                let state = self.evaluate("({connection:document.getElementById('connection-status')?.textContent,notice:document.getElementById('notice')?.textContent,history:document.getElementById('history')?.textContent})").await;
                bail!("browser condition timed out: {condition}; page state: {state:?}");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    async fn key(&mut self, key: &str) -> Result<()> {
        self.evaluate(&format!("document.getElementById('api-key').value={}; document.getElementById('connect').click()", serde_json::to_string(key)?)).await?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Observed {
    path: String,
    authorization: Option<String>,
    content_type: Option<String>,
    status: StatusCode,
}
type Requests = Arc<Mutex<Vec<Observed>>>;
fn latest(requests: &Requests, path: &str) -> Result<Observed> {
    requests
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|request| request.path == path)
        .cloned()
        .with_context(|| format!("browser did not request {path}"))
}
fn configuration_request(requests: &Requests, key: Option<&str>, status: StatusCode) -> Result<()> {
    let request = latest(requests, "/playground/config")?;
    ensure!(
        request.status == status,
        "unexpected config status: {request:?}"
    );
    ensure!(
        request.authorization == key.map(|key| format!("Bearer {key}")),
        "browser used a stale or missing credential: {request:?}"
    );
    Ok(())
}

async fn exercise(
    browser: &mut Browser,
    origin: &str,
    requests: &Requests,
    calls: &AtomicUsize,
) -> Result<()> {
    let bearer = format!("Bearer {KEY}");
    let mut cdp = browser.connect().await?;
    cdp.call("Page.enable", json!({})).await?;
    cdp.call(
        "Page.navigate",
        json!({"url":format!("{origin}/playground")}),
    )
    .await?;
    let needs_key = "document.getElementById('model')?.textContent === '需要 API 密钥' && document.getElementById('submit').disabled";
    let connected = "document.getElementById('model')?.textContent === 'declared-image-alias' && !document.getElementById('submit').disabled";
    cdp.wait_for(needs_key).await?;
    configuration_request(requests, None, StatusCode::UNAUTHORIZED)?;
    cdp.key(WRONG_KEY).await?;
    cdp.wait_for("document.getElementById('connection-status').textContent.includes('密钥无效')")
        .await?;
    configuration_request(requests, Some(WRONG_KEY), StatusCode::UNAUTHORIZED)?;
    cdp.key(KEY).await?;
    cdp.wait_for(connected).await?;
    configuration_request(requests, Some(KEY), StatusCode::OK)?;

    cdp.evaluate("document.getElementById('prompt').value='browser generation'; document.getElementById('submit').click()").await?;
    cdp.wait_for("document.getElementById('history').textContent.includes('metadata-only fixture cannot generate')").await?;
    ensure!(
        calls.load(Ordering::SeqCst) == 1,
        "generation did not reach the original image worker"
    );
    let request = latest(requests, "/v1/images/generations")?;
    ensure!(
        request.authorization.as_deref() == Some(bearer.as_str())
            && request.content_type.as_deref() == Some("application/json")
            && request.status != StatusCode::UNAUTHORIZED,
        "actual JSON generation fetch failed authentication: {request:?}"
    );

    let png = browser.profile.0.join("reference.png");
    {
        let mut encoder = png::Encoder::new(fs::File::create(&png)?, 2, 2);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&[127; 12])?;
        writer.finish()?;
    }
    cdp.evaluate("document.getElementById('mode-edit').click()")
        .await?;
    let document = cdp.call("DOM.getDocument", json!({})).await?;
    let input = cdp
        .call(
            "DOM.querySelector",
            json!({"nodeId":document["root"]["nodeId"],"selector":"#file"}),
        )
        .await?;
    cdp.call(
        "DOM.setFileInputFiles",
        json!({"nodeId":input["nodeId"],"files":[png]}),
    )
    .await?;
    cdp.wait_for("document.getElementById('reference-list').children.length === 1 && !document.getElementById('submit').disabled && document.getElementById('notice').classList.contains('success')").await?;
    cdp.evaluate("document.getElementById('prompt').value='browser edit'; document.getElementById('submit').click()").await?;
    cdp.wait_for("document.getElementById('history').textContent.includes('metadata-only fixture cannot edit')").await?;
    ensure!(
        calls.load(Ordering::SeqCst) == 2,
        "multipart edit did not reach the original image worker"
    );
    let request = latest(requests, "/v1/images/edits")?;
    ensure!(
        request.authorization.as_deref() == Some(bearer.as_str())
            && request
                .content_type
                .as_deref()
                .is_some_and(|kind| kind.starts_with("multipart/form-data; boundary="))
            && request.status != StatusCode::UNAUTHORIZED,
        "actual multipart edit fetch failed authentication: {request:?}"
    );

    cdp.key(WRONG_KEY).await?;
    cdp.wait_for(needs_key).await?;
    configuration_request(requests, Some(WRONG_KEY), StatusCode::UNAUTHORIZED)?;
    cdp.key(KEY).await?;
    cdp.wait_for(connected).await?;
    cdp.evaluate("document.getElementById('clear-key').click()")
        .await?;
    cdp.wait_for(needs_key).await?;
    configuration_request(requests, None, StatusCode::UNAUTHORIZED)?;
    ensure!(
        cdp.evaluate("document.getElementById('api-key').value")
            .await?
            == "",
        "clear left the credential input populated"
    );
    cdp.key(KEY).await?;
    cdp.wait_for(connected).await?;
    cdp.call("Page.reload", json!({"ignoreCache":true})).await?;
    cdp.wait_for(needs_key).await?;
    configuration_request(requests, None, StatusCode::UNAUTHORIZED)?;
    ensure!(cdp.evaluate("document.getElementById('api-key').value === '' && localStorage.length === 0 && sessionStorage.length === 0 && document.cookie === ''").await? == true,
        "credential survived reload in an input or browser storage");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires FERRUM_TEST_BROWSER pointing to installed Chrome/Chromium; launches an isolated headless profile"]
async fn real_browser_authenticates_json_and_multipart_without_persisting_key() -> Result<()> {
    let mut browser = Browser::launch()?;
    let (images, host, calls) = image_service_with_capacity(1, 16 << 20);
    let key = crate::ApiKey::new(KEY)?;
    let requests: Requests = Default::default();
    let capture = requests.clone();
    let router = crate::api_key::protect(
        router(AppState::default().with_images(images.clone())),
        Some(&key),
    )?
    .layer(axum::middleware::from_fn(
        move |request: Request, next: Next| {
            let capture = capture.clone();
            let mut observed = Observed {
                path: request.uri().path().to_owned(),
                authorization: request
                    .headers()
                    .get(header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned),
                content_type: request
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned),
                status: StatusCode::INTERNAL_SERVER_ERROR,
            };
            async move {
                let response = next.run(request).await;
                observed.status = response.status();
                capture.lock().unwrap().push(observed);
                response
            }
        },
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let origin = format!("http://{}", listener.local_addr()?);
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let mut server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let result = exercise(&mut browser, &origin, &requests, &calls).await;
    // Always retire the browser/socket/server and original service, including
    // assertion failures represented by the exercise Result.
    drop(browser);
    let _ = stop.send(());
    let server_result = match tokio::time::timeout(WAIT, &mut server).await {
        Ok(result) => result
            .context("browser fixture server task failed")
            .and_then(|result| result.map_err(anyhow::Error::from)),
        Err(error) => {
            server.abort();
            let _ = server.await;
            Err(anyhow::Error::from(error).context("browser fixture server did not shut down"))
        }
    };
    let retirement = images.shutdown().await;
    let owner = images.take_shutdown().await;
    result?;
    server_result?;
    retirement?;
    ensure!(
        owner.is_some_and(|owner| !owner.has_failure()),
        "image worker retained a fixture failure"
    );
    ensure!(
        host.usage()?.committed_bytes() == 0,
        "browser requests leaked Host allocations"
    );
    Ok(())
}
