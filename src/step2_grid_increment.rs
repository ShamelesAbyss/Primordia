const NEW_WORLD_WIDTH: u32 = 500;   // Wider world
const NEW_WORLD_HEIGHT: u32 = 100;  // Taller world

// Scale kernel radius based on new width
let scaled_min_radius = ((MIN_RADIUS as f32) * (NEW_WORLD_WIDTH as f32 / 278.0)).round() as i32;
let scaled_max_radius = ((MAX_RADIUS as f32) * (NEW_WORLD_WIDTH as f32 / 278.0)).round() as i32;

// ... existing usage of NEW_WORLD_WIDTH / NEW_WORLD_HEIGHT ...
