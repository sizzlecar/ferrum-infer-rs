//! Only the Vast endpoints needed for one bounded release lease.
use reqwest::{Client as HttpClient, Method};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, time::Duration};

pub(super) struct Client {
    http: HttpClient,
    base: String,
    token: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct Instance {
    pub id: u64,
    pub label: Option<String>,
    pub actual_status: Option<String>,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct Offer {
    pub id: u64,
    pub gpu_name: String,
    pub num_gpus: u64,
    pub gpu_ram: f64,
    pub cpu_ram: f64,
    pub compute_cap: u64,
    pub cpu_arch: String,
    pub driver_version: String,
    pub disk_space: f64,
    pub dph_total: f64,
    pub inet_down_cost: f64,
    pub inet_up_cost: f64,
    pub duration: f64,
    pub rentable: bool,
    pub rented: bool,
    pub verification: String,
    pub is_bid: bool,
    pub external: Option<bool>,
}

impl Offer {
    pub fn eligible(&self, args: &super::ExecuteArgs) -> bool {
        let driver: Vec<_> = self
            .driver_version
            .split('.')
            .map(str::parse::<u32>)
            .collect();
        let driver_ok = match driver.as_slice() {
            [Ok(major), Ok(minor), Ok(patch)] => (*major, *minor, *patch) >= (550, 54, 14),
            _ => false,
        };
        // Exclude anomalous 4090 listings advertising 48 GB. These are hardware
        // product identities, not model architecture/capability substitutions.
        matches!(
            self.gpu_name.as_str(),
            "RTX 6000Ada" | "RTX 6000 Ada" | "L40" | "L40S"
        ) && self.num_gpus == 1
            && self.compute_cap == 890
            && self.cpu_arch == "amd64"
            && self.gpu_ram.is_finite()
            && (45000.0..=50000.0).contains(&self.gpu_ram)
            && self.cpu_ram.is_finite()
            && self.cpu_ram >= args.min_cpu_ram_mb as f64
            && self.disk_space.is_finite()
            && self.disk_space >= args.disk_gib as f64
            && self.duration.is_finite()
            && self.duration >= args.lease_secs as f64
            && self.rentable
            && !self.rented
            && !self.is_bid
            && self.external != Some(true)
            && self.verification == "verified"
            && driver_ok
            && bounded_price(self.dph_total, args.max_hourly_usd)
            && bounded_price(self.inet_down_cost, args.max_network_usd_per_gb)
            && bounded_price(self.inet_up_cost, args.max_network_usd_per_gb)
    }
}

fn bounded_price(value: f64, limit: f64) -> bool {
    value.is_finite() && value >= 0.0 && value <= limit
}

impl Client {
    pub fn new(token: String) -> Result<Self, String> {
        if token.trim().is_empty() {
            return Err("VAST_API_KEY is empty".into());
        }
        Ok(Self {
            http: HttpClient::builder()
                .timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|_| "build Vast HTTP client")?,
            base: "https://console.vast.ai".into(),
            token,
        })
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        query: &[(&str, String)],
    ) -> Result<Value, String> {
        let mut request = self
            .http
            .request(method.clone(), format!("{}{path}", self.base))
            .bearer_auth(&self.token)
            .query(query);
        if let Some(body) = body {
            request = request.json(&body);
        }
        let mut response = request.send().await.map_err(|_| {
            format!("Vast {method} {path}: transport failure (outcome may be ambiguous)")
        })?;
        if !response.status().is_success() {
            return Err(format!(
                "Vast {method} {path}: HTTP {}",
                response.status().as_u16()
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| format!("Vast {method} {path}: incomplete response"))?
        {
            if bytes.len() + chunk.len() > 1024 * 1024 {
                return Err("Vast response exceeds 1 MiB".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes)
            .map_err(|_| format!("Vast {method} {path}: invalid JSON response"))
    }

    pub async fn offers(&self, args: &super::ExecuteArgs) -> Result<Vec<Offer>, String> {
        let value = self.request(Method::POST, "/api/v0/bundles/", Some(json!({
            "verified":{"eq":true}, "external":{"eq":false}, "rentable":{"eq":true}, "rented":{"eq":false},
            "num_gpus":{"eq":1}, "compute_cap":{"eq":890}, "cpu_arch":{"eq":"amd64"},
            "gpu_ram":{"gte":45000,"lte":50000}, "cpu_ram":{"gte":args.min_cpu_ram_mb},
            "disk_space":{"gte":args.disk_gib}, "allocated_storage":args.disk_gib,
            "duration":{"gte":args.lease_secs}, "dph_total":{"lte":args.max_hourly_usd},
            "inet_down_cost":{"lte":args.max_network_usd_per_gb}, "inet_up_cost":{"lte":args.max_network_usd_per_gb},
            "type":"on-demand", "order":[["dph_total","asc"]], "limit":5
        })), &[]).await?;
        let rows = value["offers"]
            .as_array()
            .ok_or("Vast search omitted offers")?;
        let mut offers: Vec<Offer> = rows
            .iter()
            .filter_map(|row| serde_json::from_value(row.clone()).ok())
            .filter(|offer: &Offer| offer.eligible(args))
            .collect();
        offers.sort_by(|left, right| left.dph_total.total_cmp(&right.dph_total));
        Ok(offers)
    }

    pub async fn create(
        &self,
        offer: u64,
        disk_gib: u64,
        label: &str,
        image: &str,
    ) -> Result<u64, String> {
        let value = self.request(Method::PUT, &format!("/api/v0/asks/{offer}/"), Some(json!({
            "client_id":"me", "image":image, "disk":disk_gib, "label":label,
            "runtype":"ssh_proxy", "cancel_unavail":true, "onstart":"touch /root/.no_auto_tmux"
        })), &[]).await?;
        if value["success"] != true {
            return Err(
                "Vast create did not confirm success; reconcile label before further action".into(),
            );
        }
        value["new_contract"]
            .as_u64()
            .filter(|id| *id > 0)
            .ok_or_else(|| "Vast create omitted contract ID; reconcile label".into())
    }

    pub async fn attach(&self, id: u64, public_key: &str) -> Result<(), String> {
        let value = self
            .request(
                Method::POST,
                &format!("/api/v0/instances/{id}/ssh/"),
                Some(json!({"ssh_key":public_key})),
                &[],
            )
            .await?;
        if value["success"] != true {
            return Err("Vast key attachment did not confirm success".into());
        }
        Ok(())
    }

    pub async fn instance(&self, id: u64) -> Result<Option<Instance>, String> {
        let value = self
            .request(
                Method::GET,
                &format!("/api/v0/instances/{id}/"),
                None,
                &[("owner", "me".into())],
            )
            .await?;
        let instance = value
            .get("instances")
            .ok_or("Vast single-instance response omitted instances")?;
        if instance.is_null() {
            return Ok(None);
        }
        let instance: Instance = serde_json::from_value(instance.clone())
            .map_err(|_| "invalid Vast single-instance object")?;
        if instance.id != id {
            return Err("Vast returned a different instance ID".into());
        }
        Ok(Some(instance))
    }

    pub async fn instances(&self) -> Result<Vec<Instance>, String> {
        let mut result = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen = BTreeSet::new();
        loop {
            let mut query = vec![
                ("limit", "25".into()),
                (
                    "select_cols",
                    serde_json::to_string(&["id", "label", "actual_status"])
                        .map_err(|e| e.to_string())?,
                ),
            ];
            if let Some(cursor) = &cursor {
                query.push(("after_token", cursor.clone()));
            }
            let value = self
                .request(Method::GET, "/api/v1/instances/", None, &query)
                .await?;
            let mut page: Vec<Instance> = serde_json::from_value(value["instances"].clone())
                .map_err(|_| "invalid Vast instance page")?;
            result.append(&mut page);
            match value.get("next_token") {
                None | Some(Value::Null) => return Ok(result),
                Some(Value::String(next)) if !next.is_empty() && seen.insert(next.clone()) => {
                    cursor = Some(next.clone())
                }
                _ => return Err("Vast pagination cursor invalid or repeated".into()),
            }
        }
    }

    pub async fn destroy(&self, id: u64) -> Result<(), String> {
        let value = self
            .request(
                Method::DELETE,
                &format!("/api/v0/instances/{id}/"),
                None,
                &[],
            )
            .await?;
        if value["success"] != true {
            return Err(format!("Vast destroy {id} did not confirm success"));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "api_tests.rs"]
mod tests;
