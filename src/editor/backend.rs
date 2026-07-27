/// Trait that abstracts editor rendering so the engine can run headless.
pub trait EditorBackend {
    fn begin_frame(&mut self, _ctx: &egui::Context) {}
    fn end_frame(&mut self) {}
    fn render_viewport(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
    ) {
    }
    fn is_headless(&self) -> bool {
        false
    }
}

pub struct HeadlessEditor {
    pub frame_count: u64,
}

impl HeadlessEditor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl Default for HeadlessEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorBackend for HeadlessEditor {
    fn begin_frame(&mut self, _ctx: &egui::Context) {
        self.frame_count += 1;
    }
    fn end_frame(&mut self) {}
    fn render_viewport(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
    ) {
    }
    fn is_headless(&self) -> bool {
        true
    }
}

pub struct WgpuEditor {
    pub frame_count: u64,
}

impl WgpuEditor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl Default for WgpuEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorBackend for WgpuEditor {
    fn begin_frame(&mut self, _ctx: &egui::Context) {
        self.frame_count += 1;
    }
    fn end_frame(&mut self) {}
    fn render_viewport(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
    ) {
        // EditorUi owns the egui_wgpu::Renderer and handles viewport rendering
        // through its own render() pipeline.  This method is reserved for future
        // headless/remote rendering where the viewport texture needs to be
        // re-encoded to a stream or off-screen buffer.
    }
    fn is_headless(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_editor_new() {
        let e = HeadlessEditor::new();
        assert_eq!(e.frame_count, 0);
        assert!(e.is_headless());
    }

    #[test]
    fn headless_editor_begin_frame() {
        let mut e = HeadlessEditor::new();
        let ctx = egui::Context::default();
        e.begin_frame(&ctx);
        assert_eq!(e.frame_count, 1);
        e.begin_frame(&ctx);
        assert_eq!(e.frame_count, 2);
    }

    #[test]
    fn wgpu_editor_new() {
        let e = WgpuEditor::new();
        assert_eq!(e.frame_count, 0);
        assert!(!e.is_headless());
    }

    #[test]
    fn editor_backend_trait_object() {
        let mut editors: Vec<Box<dyn EditorBackend>> = vec![
            Box::new(HeadlessEditor::new()),
            Box::new(WgpuEditor::new()),
        ];
        let ctx = egui::Context::default();
        for e in &mut editors {
            e.begin_frame(&ctx);
            e.end_frame();
        }
        assert!(editors[0].is_headless());
        assert!(!editors[1].is_headless());
    }
}
