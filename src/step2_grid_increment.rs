// === Step 2: CPU + GPU Grid Expansion ===

// --- New world dimensions ---
const NEW_WORLD_WIDTH: u32 = 350;   // Incremental width increase
const NEW_WORLD_HEIGHT: u32 = 57;   // Keep height same for now

// --- Scale kernel radius based on new width ---
let scaled_min_radius = ((MIN_RADIUS as f32) * (NEW_WORLD_WIDTH as f32 / 278.0)).round() as i32;
let scaled_max_radius = ((MAX_RADIUS as f32) * (NEW_WORLD_WIDTH as f32 / 278.0)).round() as i32;

// --- Initialize CPU World with new dimensions ---
let mut world = World::new(
    NEW_WORLD_WIDTH,
    NEW_WORLD_HEIGHT,
    scaled_min_radius..scaled_max_radius
);

// --- GPU Engine Initialization (if enabled) ---
#[cfg(feature = "gpu")]
let gpu_engine = RealLeniaGpuEngine::new(
    NEW_WORLD_WIDTH,
    NEW_WORLD_HEIGHT,
    channels,
    cell_len,
    rules_len,
    taps_len
)?;

// --- Preserve particle count ---
world.particle_count = PARTICLE_COUNT; // keep same count as before

