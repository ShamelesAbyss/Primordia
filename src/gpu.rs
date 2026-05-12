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

        let run_step_test = std::env::var("PRIMORDIA_GPU_STEP_TEST")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let compute_ok = if run_step_test {
            primordia_step_test(&adapter).await.is_ok()
        } else {
            false
        };

        GpuStatus {
            enabled: true,
            available: true,
            compute_ok,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            note: if compute_ok {
                "GPU adapter detected and Primordia-shaped step test passed.".to_string()
            } else if run_step_test {
                "GPU adapter detected, but Primordia-shaped step test failed. Falling back to Rayon CPU."
                    .to_string()
            } else {
                "GPU adapter detected. Step test skipped unless PRIMORDIA_GPU_STEP_TEST=1."
                    .to_string()
            },
        }
    })
}

#[cfg(feature = "gpu")]
async fn primordia_step_test(adapter: &wgpu::Adapter) -> Result<(), String> {
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Primordia GPU Step Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
            None,
        )
        .await
        .map_err(|err| format!("request_device failed: {err:?}"))?;

    let width: u32 = 32;
    let height: u32 = 18;
    let channels: u32 = 4;
    let len = (width * height * channels) as usize;

    let mut cells = vec![0.0_f32; len];

    for y in 7..11 {
        for x in 12..20 {
            for c in 0..channels {
                let idx = ((y * width + x) * channels + c) as usize;
                cells[idx] = 0.25 + c as f32 * 0.08;
            }
        }
    }

    let next = vec![0.0_f32; len];
    let params = [width, height, channels, len as u32];

    let cells_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Primordia GPU Cells Buffer"),
        size: (cells.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let next_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Primordia GPU Next Buffer"),
        size: (next.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Primordia GPU Params Buffer"),
        size: (params.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&cells_buffer, 0, bytemuck::cast_slice(&cells));
    queue.write_buffer(&next_buffer, 0, bytemuck::cast_slice(&next));
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Primordia GPU Step Test Shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Params {
    width: u32,
    height: u32,
    channels: u32,
    len: u32,
}

@group(0) @binding(0)
var<storage, read> cells: array<f32>;

@group(0) @binding(1)
var<storage, read_write> next_cells: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn wrap(v: i32, max_v: u32) -> u32 {
    let m = i32(max_v);
    return u32(((v % m) + m) % m);
}

fn idx(x: u32, y: u32, c: u32) -> u32 {
    return ((y * params.width + x) * params.channels + c);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        let i = idx(x, y, c);
        let old = cells[i];

        let up = wrap(i32(y) - 1, params.height);
        let down = wrap(i32(y) + 1, params.height);
        let left = wrap(i32(x) - 1, params.width);
        let right = wrap(i32(x) + 1, params.width);

        var lap = 0.0;
        lap = lap + cells[idx(x, up, c)];
        lap = lap + cells[idx(x, down, c)];
        lap = lap + cells[idx(left, y, c)];
        lap = lap + cells[idx(right, y, c)];
        lap = lap - old * 4.0;

        let growth = old * 0.985 + lap * 0.08 + 0.001;
        next_cells[i] = clamp(growth, 0.0, 1.0);
    }
}
"#
            .into(),
        ),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Primordia GPU Step Test Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Primordia GPU Step Test Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cells_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: next_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Primordia GPU Step Test Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Primordia GPU Step Test Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Primordia GPU Step Test Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Primordia GPU Step Test Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    Ok(())
}

#[cfg(not(feature = "gpu"))]
pub fn probe_gpu() -> GpuStatus {
    GpuStatus::disabled()
}
