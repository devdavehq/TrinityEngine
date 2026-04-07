use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver, Sender};

use super::mesh::Mesh;

struct LoadedMesh {
    path: String,
    result: Result<Mesh, String>,
}

pub struct MeshStreamingQueue {
    enabled: bool,
    request_tx: Option<Sender<String>>,
    result_rx: Option<Receiver<LoadedMesh>>,
    in_flight: HashSet<String>,
    pending: Vec<(String, f32)>,
    max_in_flight: usize,
}

impl MeshStreamingQueue {
    pub fn new(enabled: bool) -> Self {
        if !enabled {
            return Self {
                enabled: false,
                request_tx: None,
                result_rx: None,
                in_flight: HashSet::new(),
                pending: Vec::new(),
                max_in_flight: 0,
            };
        }

        let (req_tx, req_rx) = mpsc::channel::<String>();
        let (res_tx, res_rx) = mpsc::channel::<LoadedMesh>();

        std::thread::spawn(move || {
            while let Ok(path) = req_rx.recv() {
                let result = Mesh::load(&path);
                let _ = res_tx.send(LoadedMesh { path, result });
            }
        });

        Self {
            enabled: true,
            request_tx: Some(req_tx),
            result_rx: Some(res_rx),
            in_flight: HashSet::new(),
            pending: Vec::new(),
            max_in_flight: 2,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    #[allow(dead_code)]
    pub fn request_mesh(&mut self, path: &str) {
        self.request_mesh_with_priority(path, 0.0);
    }

    pub fn request_mesh_with_priority(&mut self, path: &str, priority: f32) {
        if !self.enabled || self.in_flight.contains(path) {
            return;
        }
        if self.pending.iter().any(|(p, _)| p == path) {
            return;
        }
        self.pending.push((path.to_string(), priority));
    }

    pub fn pump_requests(&mut self) {
        if !self.enabled {
            return;
        }
        self.pending
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        while self.in_flight.len() < self.max_in_flight && !self.pending.is_empty() {
            let (path, _) = self.pending.remove(0);
            if let Some(tx) = &self.request_tx {
                if tx.send(path.clone()).is_ok() {
                    self.in_flight.insert(path);
                }
            }
        }
    }

    pub fn poll_loaded(&mut self) -> Vec<(String, Result<Mesh, String>)> {
        let mut out = Vec::new();
        if let Some(rx) = &self.result_rx {
            while let Ok(item) = rx.try_recv() {
                self.in_flight.remove(&item.path);
                out.push((item.path, item.result));
            }
        }
        out
    }
}
