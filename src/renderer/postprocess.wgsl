struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let xy = p[vi];
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = xy * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_copy(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, in.uv);
}

@fragment
fn fs_bloom_extract(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv).rgb;
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let k = smoothstep(0.65, 1.0, luma);
    return vec4<f32>(c * k, 1.0);
}

@fragment
fn fs_blur_h(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(1.0 / tex_size.x, 0.0);
    var c = textureSample(t_src, s_src, in.uv).rgb * 0.227027;
    c += textureSample(t_src, s_src, in.uv + px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv - px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv + px * 3.230769).rgb * 0.070270;
    c += textureSample(t_src, s_src, in.uv - px * 3.230769).rgb * 0.070270;
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_blur_v(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_src, 0));
    let px = vec2<f32>(0.0, 1.0 / tex_size.y);
    var c = textureSample(t_src, s_src, in.uv).rgb * 0.227027;
    c += textureSample(t_src, s_src, in.uv + px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv - px * 1.384615).rgb * 0.316216;
    c += textureSample(t_src, s_src, in.uv + px * 3.230769).rgb * 0.070270;
    c += textureSample(t_src, s_src, in.uv - px * 3.230769).rgb * 0.070270;
    return vec4<f32>(c, 1.0);
}

@group(1) @binding(0) var t_bloom: texture_2d<f32>;
@group(1) @binding(1) var s_bloom: sampler;

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let base = textureSample(t_src, s_src, in.uv).rgb;
    let bloom = textureSample(t_bloom, s_bloom, in.uv).rgb;
    return vec4<f32>(base + bloom, 1.0);
}
