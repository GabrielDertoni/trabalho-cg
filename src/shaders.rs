use rasterization::{
    math::{
        utils::{simd_mix, simd_remap},
        Mat4x4, Vec, Vec2, Vec3, Vec4, Vec4xN,
    },
    texture::{BorrowedTextureRGBA, RowMajor, TextureWrap},
    Attributes, AttributesSimd, FragmentShaderSimd, IntoSimd, StructureOfArray, Vertex,
    VertexShader,
};

use std::simd::{LaneCount, Mask, Simd, SimdFloat, SupportedLaneCount};

use crate::LightInfo;

#[derive(Clone, Copy, Debug, IntoSimd, Attributes, AttributesSimd)]
pub struct TexturedAttributes {
    #[position]
    pub position_ndc: Vec4,
    pub frag_position: Vec3,
    pub world_normal: Vec3,
    pub uv: Vec2,
}

pub struct TexturedVertexShader {
    model: Mat4x4,
    view_projection: Mat4x4,
}

impl TexturedVertexShader {
    pub fn new(model: Mat4x4, view_projection: Mat4x4) -> Self {
        TexturedVertexShader {
            model,
            view_projection,
        }
    }
}

impl VertexShader<Vertex> for TexturedVertexShader {
    type Output = TexturedAttributes;

    fn exec(&self, vertex: Vertex) -> Self::Output {
        let world = self.model * vertex.position;
        TexturedAttributes {
            position_ndc: self.view_projection * world,
            frag_position: world.xyz(),
            world_normal: (self.model * Vec4::from((vertex.normal, 0.0))).xyz(),
            uv: vertex.uv,
        }
    }
}

pub struct TexturedFragmentShader<'a> {
    near: f32,
    far: f32,
    fog_color: Vec3,
    light_infos: &'a [LightInfo],
    ambient_light: Vec3,
    camera_pos: Vec3,
    texture: BorrowedTextureRGBA<'a, RowMajor>,
}

impl<'a> TexturedFragmentShader<'a> {
    pub fn new(
        near: f32,
        far: f32,
        fog_color: Vec3,
        texture: BorrowedTextureRGBA<'a, RowMajor>,
        camera_pos: Vec3,
        ambient_light: Vec3,
        light_infos: &'a [LightInfo],
    ) -> Self {
        TexturedFragmentShader {
            near,
            far,
            fog_color,
            texture,
            light_infos,
            ambient_light,
            camera_pos,
        }
    }
}

impl<'a, const LANES: usize> FragmentShaderSimd<TexturedAttributes, LANES>
    for TexturedFragmentShader<'a>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn exec(
        &self,
        mask: Mask<i32, LANES>,
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: TexturedAttributesSimd<LANES>,
    ) -> Vec4xN<LANES> {
        let ambient = self.ambient_light.splat();
        let mut intensity = ambient;
        for light_info in self.light_infos {
            let light_dir =
                (light_info.light.position.splat() - attrs.frag_position.xyz()).normalized();
            let diffuse_intensity = light_dir.dot(attrs.world_normal).simd_max(Simd::splat(0.));
            let diffuse = diffuse_intensity * light_info.light.color.splat();

            let view_dir = (self.camera_pos.splat() - attrs.frag_position.xyz()).normalized();
            let halfway_dir = (light_dir + view_dir).normalized();
            let specular_intensity = {
                let mut pow = attrs
                    .world_normal
                    .dot(halfway_dir)
                    .simd_max(Simd::splat(0.));
                for _ in 0..64_usize.ilog2() {
                    pow *= pow;
                }
                pow
            };

            let specular = specular_intensity * light_info.light.color.splat();
            intensity += diffuse + specular;
        }

        let texture_color = self
            .texture
            .simd_index_uv(attrs.uv, mask, TextureWrap::Repeat);
        let alpha = texture_color.w;

        let lit_color = intensity.element_mul(texture_color.xyz());

        let depth = simd_remap(
            Simd::splat(1.) / attrs.position_ndc.w,
            Simd::splat(self.near)..Simd::splat(self.far),
            Simd::splat(0.)..Simd::splat(1.),
        );
        let fog_color = self.fog_color.splat();
        (
            simd_mix(fog_color, lit_color.xyz(), depth * depth * depth),
            alpha,
        )
            .into()
    }
}
