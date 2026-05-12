#[derive(Debug, Clone)]
pub struct GpuStatus {
    pub enabled: bool,
    pub available: bool,
    pub adapter_name: String,
    pub backend: String,
    pub note: String,
}

impl GpuStatus {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            available: false,
            adapter_name: "none".to_string(),
            backend: "cpu-rayon".to_string(),
            note: "GPU feature disabled, using Rayon CPU backend".to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn label(&self) -> String {
        if self.enabled && self.available {
            format!("gpu={} backend={}", self.adapter_name, self.backend)
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
                adapter_name: "none".to_string(),
                backend: "unavailable".to_string(),
                note: "GPU feature enabled, but no adapter found. Falling back to Rayon CPU."
                    .to_string(),
            };
        };

        let info = adapter.get_info();

        GpuStatus {
            enabled: true,
            available: true,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            note: "GPU adapter detected. Compute backend will be wired in next.".to_string(),
        }
    })
}

#[cfg(not(feature = "gpu"))]
pub fn probe_gpu() -> GpuStatus {
    GpuStatus::disabled()
}
