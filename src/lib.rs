use wgpu::{
    PipelineCompilationOptions,
    RenderPipelineDescriptor,
    PipelineLayoutDescriptor,
    COPY_BUFFER_ALIGNMENT,
    VertexBufferLayout,
    DepthStencilState,
    TextureViewDescriptor,
    SamplerBindingType,
    BindingType,
    ShaderStages,
    BindGroupLayoutEntry,
    TextureSampleType,
    TextureViewDimension,
    BindGroupLayoutDescriptor,
    BindGroupLayout,
    Sampler,
    ImageDataLayout,
    TextureDescriptor,
    TextureAspect,
    Origin3d,
    ImageCopyTexture,
    TextureUsages,
    TextureDimension,
    Extent3d,
    vertex_attr_array,
    MultisampleState,
    BufferDescriptor,
    VertexAttribute,
    RenderPipeline,
    PrimitiveState,
    VertexStepMode,
    FragmentState,
    TextureFormat,
    BufferAddress,
    ShaderModule,
    BufferUsages,
    IndexFormat,
    VertexState,
    BindGroup,
    RenderPass,
    Buffer,
    Device,
    Queue,
};

use lyon_tessellation::{
    FillVertexConstructor,
    FillTessellator,
    FillOptions,
    FillBuilder,
    FillVertex,
    BuffersBuilder,
    VertexBuffers,
};

use image::RgbaImage;

use std::collections::HashMap;

type Callback = Box<dyn Fn(&mut FillBuilder) -> ((u32, u32, u32, u32), Option<RgbaImage>)>;
type Bound = (u32, u32, u32, u32);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DefaultVertex {
    position: [f32; 2],
    tx: [f32; 2],
    z: f32
}

impl DefaultVertex {
    const ATTRIBS: [VertexAttribute; 3] =
        vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32];
}

impl Vertex for DefaultVertex {
    type Constructor = DefaultVertexConstructor;

    fn constructor() -> Self::Constructor {DefaultVertexConstructor}
    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    fn shader(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"))
    }
}

#[derive(Clone)]
pub struct DefaultVertexConstructor;
impl FillVertexConstructor<DefaultVertex> for DefaultVertexConstructor {
    fn new_vertex(&mut self, mut vertex: FillVertex) -> DefaultVertex {
        let attrs: [f32; 5] = vertex.interpolated_attributes().try_into()
            .expect("Expected start(2 f32) and end(2 f32) coordinates, with a z_index(f32)");
        let start = [attrs[0], attrs[1]];
        let end = [attrs[2], attrs[3]];
        let pos = vertex.position().to_array();
        let tx = [(start[0]-pos[0])/(start[0]-end[0]), (start[1]-pos[1])/(start[1]-end[1])];
        DefaultVertex{
            position: pos,
            tx,
            z: attrs[4]
        }
    }
}

pub trait Vertex: Copy + bytemuck::Pod + bytemuck::Zeroable {
    type Constructor: FillVertexConstructor<Self>;

    fn constructor() -> Self::Constructor;
    fn layout() -> VertexBufferLayout<'static>;
    fn shader(device: &Device) -> ShaderModule;
}

pub struct ImageRenderer<V: Vertex = DefaultVertex> {
    render_pipeline: RenderPipeline,
    vertex_buffer_size: u64,
    vertex_buffer: Buffer,
    index_buffer_size: u64,
    index_buffer: Buffer,
    lyon_buffers: VertexBuffers<V, u16>,
    slices: Vec<(usize, usize, Bound, RgbaImage)>,
    images: HashMap<RgbaImage, BindGroup>,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler
}

impl<V: Vertex> ImageRenderer<V> {
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

        let shader = V::shader(device);
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
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[
                    V::layout()
                ]
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some((*texture_format).into())],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil,
            multisample,
            multiview: None,
            cache: None
        });

        let vertex_buffer_size = Self::next_copy_buffer_size(4096);
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Lyon Vertex Buffer"),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer_size = Self::next_copy_buffer_size(4096);
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Lyon Index Buffer"),
            size: index_buffer_size,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        let lyon_buffers: VertexBuffers<V, u16> = VertexBuffers::new();
        ImageRenderer{
            render_pipeline,
            vertex_buffer_size,
            vertex_buffer,
            index_buffer_size,
            index_buffer,
            lyon_buffers,
            slices: Vec::new(),
            images: HashMap::new(),
            bind_group_layout,
            sampler
        }
    }

    /// Prepare for rendering this frame; create all resources that will be
    /// used during the next render that do not already exist.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        fill_options: &FillOptions,
        callbacks: Vec<Callback>
    ) {
        self.lyon_buffers.clear();

        self.slices = Vec::new();

        let mut index = 0;

        let mut tessellator = FillTessellator::new();
        for callback in callbacks {
            let mut buffer = BuffersBuilder::new(&mut self.lyon_buffers, V::constructor());

            let mut builder = tessellator.builder_with_attributes(5, fill_options, &mut buffer);

            let (bounds, image) = callback(&mut builder);
            let image = image.unwrap();

            builder.build().unwrap();

            let buffer_len = buffer.buffers().indices.len();

            self.create_bind_group(device, queue, &image);
            self.slices.push((index, buffer_len, bounds, image));
            index = buffer_len;
        }

        if self.lyon_buffers.vertices.is_empty() || self.lyon_buffers.indices.is_empty() {return;}

        let vertices_raw = bytemuck::cast_slice(&self.lyon_buffers.vertices);
        if self.vertex_buffer_size >= vertices_raw.len() as u64 {
            Self::write_buffer(queue, &self.vertex_buffer, vertices_raw);
        } else {
            let (vertex_buffer, vertex_buffer_size) = Self::create_oversized_buffer(
                device,
                Some("Lyon Vertex Buffer"),
                vertices_raw,
                BufferUsages::VERTEX | BufferUsages::COPY_DST
            );
            self.vertex_buffer = vertex_buffer;
            self.vertex_buffer_size = vertex_buffer_size;
        }

        let indices_raw = bytemuck::cast_slice(&self.lyon_buffers.indices);
        if self.index_buffer_size >= indices_raw.len() as u64 {
            Self::write_buffer(queue, &self.index_buffer, indices_raw);
        } else {
            let (index_buffer, index_buffer_size) = Self::create_oversized_buffer(
                device,
                Some("Lyon Index Buffer"),
                indices_raw,
                BufferUsages::INDEX | BufferUsages::COPY_DST
            );
            self.index_buffer = index_buffer;
            self.index_buffer_size = index_buffer_size;
        }
    }

    /// Render using caller provided render pass.
    pub fn render(&self, render_pass: &mut RenderPass<'_>) {
        if self.lyon_buffers.vertices.is_empty() || self.lyon_buffers.indices.is_empty() {return;}

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        for (start, end, bound, image) in &self.slices {
            render_pass.set_bind_group(0, self.images.get(image).unwrap(), &[]);
            render_pass.set_scissor_rect(bound.0, bound.1, bound.2, bound.3);
            render_pass.draw_indexed(*start as u32..*end as u32, 0, 0..1);
        }
    }

    fn write_buffer(queue: &Queue, buffer: &Buffer, slice: &[u8]) {
        let pad: usize = slice.len() % 4;
        let slice = if pad != 0 {
            &[slice, &vec![0u8; pad]].concat()
        } else {slice};
        queue.write_buffer(buffer, 0, slice);
    }

    fn next_copy_buffer_size(size: u64) -> u64 {
        let align_mask = COPY_BUFFER_ALIGNMENT - 1;
        ((size.next_power_of_two() + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
    }

    fn create_oversized_buffer(
        device: &Device,
        label: Option<&str>,
        contents: &[u8],
        usage: BufferUsages,
    ) -> (Buffer, u64) {
        let size = Self::next_copy_buffer_size(contents.len() as u64);
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut()[..contents.len()].copy_from_slice(contents);
        buffer.unmap();
        (buffer, size)
    }

    fn create_bind_group(&mut self, device: &Device, queue: &Queue, image: &RgbaImage) {
        if !self.images.contains_key(image) {
            log::warn!("Building bind group");
            let dimensions = image.dimensions();
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
                ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                image,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * dimensions.0),
                    rows_per_image: Some(dimensions.1),
                },
                size
            );

            let texture_view = texture.create_view(&TextureViewDescriptor::default());

            let bind_group = device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        }
                    ],
                    label: None,
                }
            );

            self.images.insert(image.clone(), bind_group);
        }
    }
}
