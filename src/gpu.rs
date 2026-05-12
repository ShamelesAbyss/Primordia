#[derive(Debug, Clone)]
pub struct GpuStatus {
    pub enabled: bool,
    pub available: bool,
    pub compute_ok: bool,
    pub adapter_name: String,
    pub backend: String,
    pub note: String,
}

impl GpuStatus {
    #[allow(dead_code)]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            available: false,
            compute_ok: false,
            adapter_name: "none".to_string(),
            backend: "cpu-rayon".to_string(),
            note: "GPU feature disabled, using Rayon CPU backend".to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn label(&self) -> String {
        if self.enabled && self.available && self.compute_ok {
            format!(
                "gpu={} backend={} compute=ok",
                self.adapter_name, self.backend
            )
        } else if self.enabled && self.available {
            format!(
                "gpu={} backend={} compute=detected-not-forced",
                self.adapter_name, self.backend
            )
        } else {
            self.note.clone()
        }
    }
}

#[cfg(feature = "gpu")]
pub fn probe_gpu() -> GpuStatus {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();

        let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        else {
            return GpuStatus {
                enabled: true,
                available: false,
                compute_ok: false,
                adapter_name: "none".to_string(),
                backend: "unavailable".to_string(),
                note: "GPU feature enabled, but no adapter found. Falling back to Rayon CPU."
                    .to_string(),
            };
        };

        let info = adapter.get_info();

        let compute_ok = std::env::var("PRIMORDIA_GPU_SMOKE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        GpuStatus {
            enabled: true,
            available: true,
            compute_ok,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            note: if compute_ok {
                "GPU adapter detected. Smoke test requested externally.".to_string()
            } else {
                "GPU adapter detected. Compute smoke test skipped unless PRIMORDIA_GPU_SMOKE=1."
                    .to_string()
            },
        }
    })
}

#[cfg(not(feature = "gpu"))]
pub fn probe_gpu() -> GpuStatus {
    GpuStatus::disabled()
}
