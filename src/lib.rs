use wgpu::{PipelineCompilationOptions, BindGroupLayoutDescriptor, RenderPipelineDescriptor, PipelineLayoutDescriptor, TextureViewDescriptor, TextureViewDimension, BindGroupLayoutEntry, SamplerBindingType, VertexBufferLayout, vertex_attr_array, DepthStencilState, TextureSampleType, TextureDescriptor, TextureDimension, TexelCopyTextureInfo, MultisampleState, VertexAttribute, BindGroupLayout, TexelCopyBufferLayout, RenderPipeline, PrimitiveState, VertexStepMode, FragmentState, TextureFormat, BufferAddress, TextureUsages, TextureAspect, ShaderStages, BufferUsages, IndexFormat, VertexState, BindingType, RenderPass, BindGroup, Origin3d, Extent3d, Sampler, Device, Queue};

use wgpu_dyn_buffer::{DynamicBufferDescriptor, DynamicBuffer};

use cyat::{VertexBuffers, ShapeBuilder, Vertex};

pub use image;
use image::RgbaImage;

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use std::hash::{DefaultHasher, Hasher, Hash};
use std::sync::Arc;

type Bound = (u32, u32, u32, u32);
pub type ImageKey = u64;

pub struct Image(pub ShapeBuilder<ImageAttributes>, pub Bound, pub ImageKey);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImageAttributes {
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub z: f32
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageVertex {
    position: [f32; 2],
    tx: [f32; 2],
    z: f32
}

impl ImageVertex {
    const ATTRIBS: [VertexAttribute; 3] =
        vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32];

    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

impl Vertex for ImageVertex {
    type Attributes = ImageAttributes;

    fn construct(pos: [f32; 2], a: Self::Attributes) -> ImageVertex {
        ImageVertex{
            position: pos,
            tx: [(a.start[0]-pos[0])/(a.start[0]-a.end[0]), (a.start[1]-pos[1])/(a.start[1]-a.end[1])],
            z: a.z
        }
    }
}

#[derive(Default)]
pub struct ImageAtlas {
    uncached: HashMap<ImageKey, RgbaImage>,
    cached: HashMap<ImageKey, Arc<BindGroup>>
}

impl ImageAtlas {
    pub fn new() -> Self {
        ImageAtlas{
            uncached: HashMap::new(),
            cached: HashMap::new(),
        }
    }

    pub fn add(&mut self, image: RgbaImage) -> ImageKey {
        let mut hasher = DefaultHasher::new();
        image.hash(&mut hasher);
        let key = hasher.finish();
        self.uncached.insert(key, image);
        key
    }

    pub fn remove(&mut self, key: &ImageKey) {
        self.uncached.remove(key);
        self.cached.remove(key);
    }

    pub fn contains(&self, key: &ImageKey) -> bool {
        self.uncached.contains_key(key) || self.cached.contains_key(key)
    }

    fn prepare(
        &mut self,
        queue: &Queue,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler
    ) {
        self.uncached.drain().collect::<Vec<_>>().into_iter().for_each(|(key, image)| {
            if let Entry::Vacant(entry) = self.cached.entry(key) {
                let mut dimensions = image.dimensions();
                dimensions.0 = dimensions.0.min(dimensions.1);
                dimensions.1 = dimensions.0.min(dimensions.1);
                let size = Extent3d {
                    width: dimensions.0,
                    height: dimensions.1,
                    depth_or_array_layers: 1,
                };

                let texture = device.create_texture(
                    &TextureDescriptor {
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::Rgba8UnormSrgb,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                        label: None,
                        view_formats: &[],
                    }
                );

                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &image,
                    TexelCopyBufferLayout{
                        offset: 0,
                        bytes_per_row: Some(4 * dimensions.0),
                        rows_per_image: Some(dimensions.1),
                    },
                    size
                );

                let texture_view = texture.create_view(&TextureViewDescriptor::default());

                let bind_group = Arc::new(device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            }
                        ],
                        label: None,
                    }
                ));
                entry.insert(bind_group);
            }
        })
    }

    fn get(&self, key: &ImageKey) -> Arc<BindGroup> {
        self.cached.get(key).expect("Image not found for ImageKey").clone()
    }
}

pub struct ImageRenderer {
    render_pipeline: RenderPipeline,
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
    cyat_buffers: VertexBuffers<ImageVertex, u16>,
    image_buffer: Vec<(usize, usize, Bound, Arc<BindGroup>)>,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler
}

impl ImageRenderer {
    /// Create all unchanging resources here.
    pub fn new(
        device: &Device,
        texture_format: &TextureFormat,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float{filterable: true},
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[ImageVertex::layout()]
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some((*texture_format).into())],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil,
            multisample,
            multiview: None,
            cache: None
        });

        let vertex_buffer = DynamicBuffer::new(device, &DynamicBufferDescriptor {
            label: None,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        let index_buffer = DynamicBuffer::new(device, &DynamicBufferDescriptor {
            label: None,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        ImageRenderer{
            render_pipeline,
            vertex_buffer,
            index_buffer,
            cyat_buffers: VertexBuffers::new(),
            image_buffer: Vec::new(),
            bind_group_layout,
            sampler
        }
    }

    /// Prepare for rendering this frame; create all resources that will be
    /// used during the next render that do not already exist.
    pub fn prepare(
        &mut self,
        queue: &Queue,
        device: &Device,
        image_atlas: &mut ImageAtlas,
        images: Vec<Image>
    ) {
        self.cyat_buffers.clear();
        self.image_buffer.clear();

        image_atlas.prepare(queue, device, &self.bind_group_layout, &self.sampler);

        let mut index = 0;

        for Image(shape, bound, key) in images {
            shape.build(&mut self.cyat_buffers);
            let buffer_len = self.cyat_buffers.indices.len();
            self.image_buffer.push((index, buffer_len, bound, image_atlas.get(&key)));
            index = buffer_len;
        }

        if self.cyat_buffers.vertices.is_empty() || self.cyat_buffers.indices.is_empty() {return;}

        self.vertex_buffer.write_buffer(device, queue, bytemuck::cast_slice(&self.cyat_buffers.vertices));
        self.index_buffer.write_buffer(device, queue, bytemuck::cast_slice(&self.cyat_buffers.indices));
    }

    /// Render using caller provided render pass.
    pub fn render(&self, render_pass: &mut RenderPass<'_>) {
        if self.cyat_buffers.vertices.is_empty() || self.cyat_buffers.indices.is_empty() {return;}

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.as_ref().slice(..));
        render_pass.set_index_buffer(self.index_buffer.as_ref().slice(..), IndexFormat::Uint16);
        for (start, end, bound, bind_group) in &self.image_buffer {
            render_pass.set_bind_group(0, Some(&**bind_group), &[]);
            render_pass.set_scissor_rect(bound.0, bound.1, bound.2, bound.3);
            render_pass.draw_indexed(*start as u32..*end as u32, 0, 0..1);
        }
    }
}
