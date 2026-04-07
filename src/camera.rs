// glam is our math library.
// Mat4 = a 4×4 matrix. Used for all transforms in 3D graphics.
// Vec3 = a 3-element vector (x, y, z).
// These types also implement bytemuck traits so we can send them to the GPU.
use glam::{Mat4, Vec3};

// Camera2D represents a 2D camera looking down at a flat world.
// It has a position in the world and a zoom level.
// "2D" means we use orthographic projection — no perspective distortion.
pub struct Camera2D {
    // Where the camera is in world space.
    // Moving this pans the view.
    pub position: Vec3,

    // How "zoomed in" the camera is.
    // 1.0 = normal. 2.0 = zoomed in (things appear twice as big).
    // 0.5 = zoomed out (things appear half as big, more world visible).
    pub zoom: f32,

    // The aspect ratio of the window: width / height.
    // A 1280×720 window has aspect = 1280/720 ≈ 1.777.
    // We need this so the projection doesn't stretch things.
    pub aspect: f32,
}

impl Camera2D {
    // Create a new camera at the world origin, not zoomed.
    // aspect: pass window_width / window_height when creating.
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::ZERO, // Vec3::ZERO = (0.0, 0.0, 0.0)
            zoom:     1.0,
            aspect,
        }
    }

    // view_projection_matrix() computes the combined matrix the GPU needs.
    //
    // Why one matrix instead of two?
    // The GPU applies one matrix per vertex. We could send two and multiply
    // them in the shader, but it's more efficient to pre-multiply on the CPU
    // once per frame and send the result. Same output, less GPU work.
    pub fn view_projection_matrix(&self) -> Mat4 {
        // ── View matrix ───────────────────────────────────────────────────
        // The view matrix moves the world so the camera appears to be at
        // the origin looking forward. It's the inverse of where the camera is.
        //
        // Mat4::from_translation() creates a matrix that shifts everything
        // by the given vector. We negate position because: if the camera
        // moves right (+x), the world appears to move left (-x) in view space.
        let view = Mat4::from_translation(-self.position);

        // ── Projection matrix ─────────────────────────────────────────────
        // Orthographic projection: no perspective distortion.
        // Things at any distance appear the same size.
        // Perfect for 2D games and top-down views.
        //
        // orthographic_rh() = right-handed coordinate system.
        // The six parameters define a rectangular view volume (frustum):
        //   left, right: horizontal extent
        //   bottom, top: vertical extent
        //   near, far:   depth range (we use -1 to +1 for 2D)
        //
        // We divide by zoom: bigger zoom → smaller extents → things look bigger.
        // We multiply by aspect on x: a wider screen shows more world horizontally.
        let half_h = 1.0 / self.zoom;         // how much world fits vertically
        let half_w = half_h * self.aspect;    // how much world fits horizontally

        let proj = Mat4::orthographic_rh(
            -half_w,  // left edge of visible world
             half_w,  // right edge
            -half_h,  // bottom edge
             half_h,  // top edge
            -1.0,     // near plane (anything closer is clipped)
             1.0,     // far plane  (anything farther is clipped)
        );

        // Multiply projection × view.
        // Matrix multiplication order matters: right-to-left.
        // This means: first apply view, then apply projection.
        // The result is one matrix that does both transforms at once.
        proj * view
    }
}

pub struct Camera3D {
    // Where the camera is in 3D world space.
    pub position: glam::Vec3,

    // What the camera is looking at.
    pub target: glam::Vec3,

    // Which direction is "up" — almost always (0, 1, 0).
    pub up: glam::Vec3,

    // Vertical field of view in degrees.
    // 60° feels natural. Wide angle = more visible, fisheye feel.
    pub fov_degrees: f32,

    // Window aspect ratio — same as Camera2D.
    pub aspect: f32,

    // Near clip plane — fragments closer than this are invisible.
    // Should be as large as possible without clipping desired objects.
    // 0.1 world units is a safe default.
    pub near: f32,

    // Far clip plane — fragments farther than this are invisible.
    // Should be as small as possible for depth precision.
    pub far: f32,
}

impl Camera3D {
    pub fn new(aspect: f32) -> Self {
        Self {
            // Start behind and above the origin, looking at it.
            position:    glam::Vec3::new(0.0, 2.0, 5.0),
            target:      glam::Vec3::ZERO,
            up:          glam::Vec3::Y,  // Vec3::Y = (0, 1, 0)
            fov_degrees: 60.0,
            aspect,
            near:        0.1,
            far:         100.0,
        }
    }

    pub fn view_projection_matrix(&self) -> glam::Mat4 {
        // View matrix: look_at_rh() builds a matrix that transforms
        // world space so the camera is at the origin looking forward.
        // "rh" = right-handed coordinate system (standard in OpenGL/wgpu).
        let view = glam::Mat4::look_at_rh(
            self.position,
            self.target,
            self.up,
        );

        // Perspective projection matrix.
        // perspective_rh() takes:
        //   fov in radians — .to_radians() converts from degrees
        //   aspect ratio — width / height
        //   near clip distance
        //   far clip distance
        let proj = glam::Mat4::perspective_rh(
            self.fov_degrees.to_radians(),
            self.aspect,
            self.near,
            self.far,
        );

        // Projection × view = full transform from world to clip space.
        proj * view
    }
}


pub trait Camera {
    fn view_projection_matrix(&self) -> glam::Mat4;
    fn position(&self) -> glam::Vec3;
}

impl Camera for Camera2D {
    fn view_projection_matrix(&self) -> glam::Mat4 { self.view_projection_matrix() }
    fn position(&self) -> glam::Vec3 { self.position }
}

impl Camera for Camera3D {
    fn view_projection_matrix(&self) -> glam::Mat4 { self.view_projection_matrix() }
    fn position(&self) -> glam::Vec3 { self.position }
}