// === Step 1: Incremental Width Expansion ===

const NEW_WIDTH: u32 = 350;
const NEW_HEIGHT: u32 = 57;

// Scale kernel radius proportionally to width
let kernel_radius_min = ((MIN_RADIUS as f32) * (NEW_WIDTH as f32 / 278.0)).round() as i32;
let kernel_radius_max = ((MAX_RADIUS as f32) * (NEW_WIDTH as f32 / 278.0)).round() as i32;

// World initialization with new width and scaled kernel radius
let world = World::new(
    NEW_WIDTH,
    NEW_HEIGHT,
    kernel_radius_min..kernel_radius_max
);

// GPU engine (if enabled)
#[cfg(feature = "gpu")]
let gpu_engine = RealLeniaGpuEngine::new(
    NEW_WIDTH,
    NEW_HEIGHT,
    channels,
    cell_len,
    rules_len,
    taps_len
)?;
