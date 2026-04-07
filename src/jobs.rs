use rayon::{ThreadPool, ThreadPoolBuilder};

pub struct JobSystem {
    enabled: bool,
    pool: Option<ThreadPool>,
}

impl JobSystem {
    pub fn new(enabled: bool, worker_threads: usize) -> Self {
        if !enabled {
            return Self {
                enabled: false,
                pool: None,
            };
        }

        let mut builder = ThreadPoolBuilder::new();
        if worker_threads > 0 {
            builder = builder.num_threads(worker_threads);
        }

        let pool = builder.build().ok();
        let enabled = pool.is_some();
        Self { enabled, pool }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        if let Some(pool) = &self.pool {
            pool.install(f)
        } else {
            f()
        }
    }
}
