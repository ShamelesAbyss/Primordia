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
pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    layout: wgpu::BindGroupLayout,
    cells_buffer: wgpu::Buffer,
    next_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    channels: u32,
    len: usize,
}

#[cfg(feature = "gpu")]
impl GpuEngine {
    pub async fn new(width: u32, height: u32, channels: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "no GPU adapter found".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Primordia GPU Engine Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|err| format!("request_device failed: {err:?}"))?;

        let len = (width * height * channels) as usize;
        let buffer_size = (len * std::mem::size_of::<f32>()) as u64;

        let cells_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia GPU Engine Cells"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia GPU Engine Next"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia GPU Engine Params"),
            size: (4 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Primordia GPU Engine Shader"),
            source: wgpu::ShaderSource::Wgsl(PRIMORDIA_STEP_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Primordia GPU Engine Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Primordia GPU Engine Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Primordia GPU Engine Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            cells_buffer,
            next_buffer,
            params_buffer,
            width,
            height,
            channels,
            len,
        })
    }

    pub fn run_one_step(&self, cells: &[f32]) -> Result<(), String> {
        if cells.len() != self.len {
            return Err(format!(
                "cell length mismatch: got {}, expected {}",
                cells.len(),
                self.len
            ));
        }

        let params = [self.width, self.height, self.channels, self.len as u32];

        self.queue
            .write_buffer(&self.cells_buffer, 0, bytemuck::cast_slice(cells));
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Primordia GPU Engine Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Primordia GPU Engine Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Primordia GPU Engine Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        Ok(())
    }
    #[allow(dead_code)]
    pub fn run_one_step_readback(&self, cells: &[f32]) -> Result<Vec<f32>, String> {
        if cells.len() != self.len {
            return Err(format!(
                "cell length mismatch: got {}, expected {}",
                cells.len(),
                self.len
            ));
        }

        let params = [self.width, self.height, self.channels, self.len as u32];
        let byte_size = (self.len * std::mem::size_of::<f32>()) as u64;

        self.queue
            .write_buffer(&self.cells_buffer, 0, bytemuck::cast_slice(cells));
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia GPU Readback Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Primordia GPU Readback Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Primordia GPU Readback Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Primordia GPU Readback Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }

        encoder.copy_buffer_to_buffer(&self.next_buffer, 0, &readback_buffer, 0, byte_size);

        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|err| format!("GPU readback receive failed: {err:?}"))?
            .map_err(|err| format!("GPU readback map failed: {err:?}"))?;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();

        drop(mapped);
        readback_buffer.unmap();

        Ok(values)
    }
}

#[cfg(feature = "gpu")]
const PRIMORDIA_STEP_WGSL: &str = r#"
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
"#;

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
            match GpuEngine::new(32, 18, 4).await {
                Ok(engine) => {
                    let len = (32 * 18 * 4) as usize;
                    let mut cells = vec![0.0_f32; len];

                    for y in 7..11 {
                        for x in 12..20 {
                            for c in 0..4 {
                                let idx = ((y * 32 + x) * 4 + c) as usize;
                                cells[idx] = 0.25 + c as f32 * 0.08;
                            }
                        }
                    }

                    engine.run_one_step(&cells).is_ok()
                }
                Err(_) => false,
            }
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
                "GPU adapter detected and persistent engine step test passed.".to_string()
            } else if run_step_test {
                "GPU adapter detected, but persistent engine step test failed. Falling back to Rayon CPU."
                    .to_string()
            } else {
                "GPU adapter detected. Persistent engine test skipped unless PRIMORDIA_GPU_STEP_TEST=1."
                    .to_string()
            },
        }
    })
}

#[cfg(not(feature = "gpu"))]
pub fn probe_gpu() -> GpuStatus {
    GpuStatus::disabled()
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn real_world_bridge_test(
    width: u32,
    height: u32,
    channels: u32,
    cells: &[f32],
) -> Result<String, String> {
    pollster::block_on(async {
        let engine = GpuEngine::new(width, height, channels).await?;
        engine.run_one_step(cells)?;

        Ok(format!(
            "GPU bridge accepted real world buffer {}x{}x{} cells={}",
            width,
            height,
            channels,
            cells.len()
        ))
    })
}

#[cfg(not(feature = "gpu"))]
#[allow(dead_code)]
pub fn real_world_bridge_test(
    _width: u32,
    _height: u32,
    _channels: u32,
    _cells: &[f32],
) -> Result<String, String> {
    Err("GPU feature disabled".to_string())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
#[allow(dead_code)]
pub fn live_step_readback(
    width: u32,
    height: u32,
    channels: u32,
    cells: &[f32],
) -> Result<Vec<f32>, String> {
    pollster::block_on(async {
        let engine = GpuEngine::new(width, height, channels).await?;
        engine.run_one_step_readback(cells)
    })
}

#[cfg(not(feature = "gpu"))]
#[allow(dead_code)]
#[allow(dead_code)]
pub fn live_step_readback(
    _width: u32,
    _height: u32,
    _channels: u32,
    _cells: &[f32],
) -> Result<Vec<f32>, String> {
    Err("GPU feature disabled".to_string())
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRuleData {
    pub from: u32,
    pub to: u32,
    pub tap_start: u32,
    pub tap_count: u32,
    pub mu: f32,
    pub sigma: f32,
    pub weight: f32,
    pub _pad: f32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTapData {
    pub dx: i32,
    pub dy: i32,
    pub weight: f32,
    pub _pad: f32,
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn live_lenia_step_readback(
    width: u32,
    height: u32,
    channels: u32,
    cells: &[f32],
    rules: &[GpuRuleData],
    taps: &[GpuTapData],
) -> Result<Vec<f32>, String> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "no GPU adapter found".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Primordia Real Lenia GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|err| format!("request_device failed: {err:?}"))?;

        let len = cells.len();
        let cell_bytes = (len * std::mem::size_of::<f32>()) as u64;
        let rule_bytes = (rules.len().max(1) * std::mem::size_of::<GpuRuleData>()) as u64;
        let tap_bytes = (taps.len().max(1) * std::mem::size_of::<GpuTapData>()) as u64;

        let cells_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Real Cells"),
            size: cell_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let next_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Real Next"),
            size: cell_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rules_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Real Rules"),
            size: rule_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let taps_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Real Taps"),
            size: tap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = [
            width,
            height,
            channels,
            len as u32,
            rules.len() as u32,
            taps.len() as u32,
            0,
            0,
        ];

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Real Params"),
            size: (params.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Real Readback"),
            size: cell_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&cells_buffer, 0, bytemuck::cast_slice(cells));
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

        if !rules.is_empty() {
            queue.write_buffer(&rules_buffer, 0, bytemuck::cast_slice(rules));
        }

        if !taps.is_empty() {
            queue.write_buffer(&taps_buffer, 0, bytemuck::cast_slice(taps));
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Primordia Real Lenia Shader"),
            source: wgpu::ShaderSource::Wgsl(REAL_LENIA_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Primordia Real Lenia Layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Primordia Real Lenia Bind Group"),
            layout: &layout,
            entries: &[
                bind_entry(0, &cells_buffer),
                bind_entry(1, &next_buffer),
                bind_entry(2, &rules_buffer),
                bind_entry(3, &taps_buffer),
                bind_entry(4, &params_buffer),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Primordia Real Lenia Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Primordia Real Lenia Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Primordia Real Lenia Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Primordia Real Lenia Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }

        encoder.copy_buffer_to_buffer(&next_buffer, 0, &readback_buffer, 0, cell_bytes);
        queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|err| format!("GPU readback receive failed: {err:?}"))?
            .map_err(|err| format!("GPU readback map failed: {err:?}"))?;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();

        Ok(values)
    })
}

#[cfg(feature = "gpu")]
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "gpu")]
fn bind_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(feature = "gpu")]
const REAL_LENIA_WGSL: &str = r#"
struct Rule {
    from: u32,
    to: u32,
    tap_start: u32,
    tap_count: u32,
    mu: f32,
    sigma: f32,
    weight: f32,
    pad: f32,
}

struct Tap {
    dx: i32,
    dy: i32,
    weight: f32,
    pad: f32,
}

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    len: u32,
    rule_count: u32,
    tap_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0)
var<storage, read> cells: array<f32>;

@group(0) @binding(1)
var<storage, read_write> next_cells: array<f32>;

@group(0) @binding(2)
var<storage, read> rules: array<Rule>;

@group(0) @binding(3)
var<storage, read> taps: array<Tap>;

@group(0) @binding(4)
var<uniform> params: Params;

fn wrap(v: i32, max_v: u32) -> u32 {
    let m = i32(max_v);
    return u32(((v % m) + m) % m);
}

fn idx(x: u32, y: u32, c: u32) -> u32 {
    return ((y * params.width + x) * params.channels + c);
}

fn bell(x: f32, mu: f32, sigma: f32) -> f32 {
    let d = x - mu;
    return exp(-(d * d) / (2.0 * sigma * sigma));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        let out_i = idx(x, y, c);
        let old = cells[out_i];
        var delta = 0.0;

        for (var r: u32 = 0u; r < params.rule_count; r = r + 1u) {
            let rule = rules[r];

            if (rule.to != c) {
                continue;
            }

            var conv = 0.0;

            for (var t: u32 = 0u; t < rule.tap_count; t = t + 1u) {
                let tap = taps[rule.tap_start + t];

                let sx = wrap(i32(x) + tap.dx, params.width);
                let sy = wrap(i32(y) + tap.dy, params.height);
                let source_i = idx(sx, sy, rule.from);

                conv = conv + cells[source_i] * tap.weight;
            }

            let growth = bell(conv, rule.mu, rule.sigma) * 2.0 - 1.0;
            delta = delta + growth * rule.weight;
        }

        next_cells[out_i] = clamp(old + delta * 0.08, 0.0, 1.0);
    }
}
"#;

#[cfg(feature = "gpu")]
pub struct RealLeniaGpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    cells_buffer: wgpu::Buffer,
    next_buffer: wgpu::Buffer,
    rules_buffer: wgpu::Buffer,
    taps_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    channels: u32,
    cell_len: usize,
    rule_len: usize,
    tap_len: usize,
}

#[cfg(feature = "gpu")]
impl RealLeniaGpuEngine {
    pub fn new(
        width: u32,
        height: u32,
        channels: u32,
        cell_len: usize,
        rule_len: usize,
        tap_len: usize,
    ) -> Result<Self, String> {
        pollster::block_on(Self::new_async(
            width, height, channels, cell_len, rule_len, tap_len,
        ))
    }

    async fn new_async(
        width: u32,
        height: u32,
        channels: u32,
        cell_len: usize,
        rule_len: usize,
        tap_len: usize,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "no GPU adapter found".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Primordia Persistent Real Lenia Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|err| format!("request_device failed: {err:?}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Primordia Persistent Real Lenia Shader"),
            source: wgpu::ShaderSource::Wgsl(REAL_LENIA_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Primordia Persistent Real Lenia Layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Primordia Persistent Real Lenia Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Primordia Persistent Real Lenia Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cell_bytes = (cell_len * std::mem::size_of::<f32>()) as u64;
        let rule_bytes = (rule_len.max(1) * std::mem::size_of::<GpuRuleData>()) as u64;
        let tap_bytes = (tap_len.max(1) * std::mem::size_of::<GpuTapData>()) as u64;

        let cells_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Persistent Cells"),
            size: cell_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let next_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Persistent Next"),
            size: cell_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rules_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Persistent Rules"),
            size: rule_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let taps_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Persistent Taps"),
            size: tap_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Persistent Params"),
            size: (8 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primordia Persistent Readback"),
            size: cell_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Primordia Persistent Real Lenia Bind Group"),
            layout: &layout,
            entries: &[
                bind_entry(0, &cells_buffer),
                bind_entry(1, &next_buffer),
                bind_entry(2, &rules_buffer),
                bind_entry(3, &taps_buffer),
                bind_entry(4, &params_buffer),
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            bind_group,
            cells_buffer,
            next_buffer,
            rules_buffer,
            taps_buffer,
            params_buffer,
            readback_buffer,
            width,
            height,
            channels,
            cell_len,
            rule_len,
            tap_len,
        })
    }

    pub fn matches(
        &self,
        width: u32,
        height: u32,
        channels: u32,
        cell_len: usize,
        rule_len: usize,
        tap_len: usize,
    ) -> bool {
        self.width == width
            && self.height == height
            && self.channels == channels
            && self.cell_len == cell_len
            && self.rule_len == rule_len
            && self.tap_len == tap_len
    }

    pub fn step(
        &mut self,
        cells: &[f32],
        rules: &[GpuRuleData],
        taps: &[GpuTapData],
    ) -> Result<Vec<f32>, String> {
        if cells.len() != self.cell_len {
            return Err(format!(
                "cell length mismatch: got {}, expected {}",
                cells.len(),
                self.cell_len
            ));
        }

        let params = [
            self.width,
            self.height,
            self.channels,
            self.cell_len as u32,
            rules.len() as u32,
            taps.len() as u32,
            0,
            0,
        ];

        self.queue
            .write_buffer(&self.cells_buffer, 0, bytemuck::cast_slice(cells));
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        if !rules.is_empty() {
            self.queue
                .write_buffer(&self.rules_buffer, 0, bytemuck::cast_slice(rules));
        }

        if !taps.is_empty() {
            self.queue
                .write_buffer(&self.taps_buffer, 0, bytemuck::cast_slice(taps));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Primordia Persistent Real Lenia Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Primordia Persistent Real Lenia Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }

        let byte_size = (self.cell_len * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.next_buffer, 0, &self.readback_buffer, 0, byte_size);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|err| format!("GPU readback receive failed: {err:?}"))?
            .map_err(|err| format!("GPU readback map failed: {err:?}"))?;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();

        drop(mapped);
        self.readback_buffer.unmap();

        Ok(values)
    }
}
