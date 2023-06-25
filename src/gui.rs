use rasterization::{config::CullingMode, math::Vec3};

use crate::{Selected, World};

impl raster_egui::App for World {
    fn update(&mut self, ctx: &egui::Context) {
        if let Some(selected) = self.selected {
            let model_matrix = match selected {
                Selected::Model(i) => self.models[i].model_matrix().cols(),
                Selected::Light(i) => self.lights[i].model_matrix().cols(),
            };
            let gizmo = egui_gizmo::Gizmo::new("selection")
                .view_matrix(self.camera.view_matrix().cols())
                .projection_matrix(
                    self.camera
                        .projection_matrix(self.rendering_cfg.near, self.rendering_cfg.far)
                        .cols(),
                )
                .model_matrix(model_matrix)
                .mode(self.selection_mode)
                .visuals(egui_gizmo::GizmoVisuals {
                    inactive_alpha: 0.8,
                    highlight_alpha: 1.0,
                    gizmo_size: 100.0,
                    ..Default::default()
                });

            let mut ui = egui::Ui::new(
                ctx.clone(),
                egui::LayerId::background(),
                egui::Id::new("gizmo"),
                egui::Rect::EVERYTHING,
                ctx.available_rect(),
            );
            if !self.is_in_play_mode() {
                if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::W)) {
                    self.selection_mode = egui_gizmo::GizmoMode::Translate;
                }
                if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::R)) {
                    self.selection_mode = egui_gizmo::GizmoMode::Rotate;
                }
                if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::S)) {
                    self.selection_mode = egui_gizmo::GizmoMode::Scale;
                }
            }
            if let Some(response) = gizmo.interact(&mut ui) {
                let (x, y, z) = response.rotation.to_euler(glam::EulerRot::XYZ);
                match selected {
                    Selected::Model(i) => {
                        self.models[i].rotation = Vec3::from([x, y, z]);
                        self.models[i].position = Vec3::from(response.translation.to_array());
                        self.models[i].scale = Vec3::from(response.scale.to_array());
                    }
                    Selected::Light(i) => {
                        self.lights[i].light.position = Vec3::from(response.translation.to_array());
                    }
                }
            }
        }
        if self.is_in_play_mode() {
            return;
        }
        egui::Window::new("Info")
            .resizable(true)
            .constrain(false)
            .frame(
                egui::Frame::window(&ctx.style())
                    .shadow(egui::epaint::Shadow::NONE)
                    .rounding(egui::epaint::Rounding::none()),
            )
            .drag_bounds(egui::Rect::EVERYTHING)
            .show(ctx, |ui| {
                ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);

                ui.label("Camera");
                let pos = self.camera.position;
                ui.label(format!(
                    "position {{ x: {:.1}, y: {:.1}, z: {:.1} }}",
                    pos.x, pos.y, pos.z
                ));
                ui.label(format!("pitch {:.1}", self.camera.pitch));
                ui.label(format!("yaw {:.1}", self.camera.yaw));
                ui.separator();

                ui.label("Rendering stats");
                ui.label(format!("{:6.1} FPS", self.fps_counter.mean_fps()));
                ui.label(format!("{:6.1?} per frame", self.fps_counter.mean_time()));

                ui.collapsing("Options", |ui| {
                    egui::Grid::new("rendering options").show(ui, |ui| {
                        ui.label("Render distance");
                        ui.add(egui::Slider::new(&mut self.rendering_cfg.far, 1.0..=100.0));
                        ui.end_row();

                        ui.label("Fog color");
                        ui.color_edit_button_rgb(
                            self.rendering_cfg.fog_color.xyz_mut().as_mut_array(),
                        );
                        ui.end_row();

                        ui.label("Culling");
                        egui::ComboBox::from_id_source("culling")
                            .selected_text(self.rendering_cfg.culling_mode.to_string())
                            .show_ui(ui, |ui| {
                                for mode in CullingMode::enumerate() {
                                    ui.selectable_value(
                                        &mut self.rendering_cfg.culling_mode,
                                        mode,
                                        mode.to_string(),
                                    );
                                }
                            });
                        ui.end_row();

                        ui.label("Alpha clip");
                        let mut checked = self.rendering_cfg.alpha_clip.is_some();
                        ui.checkbox(&mut checked, "enable");
                        if checked {
                            if self.rendering_cfg.alpha_clip.is_none() {
                                self.rendering_cfg.alpha_clip.replace(0.0);
                            }
                            let alpha_clip = self.rendering_cfg.alpha_clip.as_mut().unwrap();
                            ui.add(egui::Slider::new(alpha_clip, 0.0..=1.0));
                        } else {
                            if self.rendering_cfg.alpha_clip.is_some() {
                                self.rendering_cfg.alpha_clip = None;
                            }
                        }
                    });
                });

                ui.collapsing("Scene", |ui| {
                    ui.label("Models");
                    for (i, model) in self.models.iter_mut().enumerate() {
                        ui.radio_value(&mut self.selected, Some(Selected::Model(i)), &model.name);
                    }

                    ui.label("Lights");
                    for (i, light) in self.lights.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.radio_value(
                                &mut self.selected,
                                Some(Selected::Light(i)),
                                &light.light.name,
                            );
                            ui.collapsing("more", |ui| {
                                ui.label("Color");
                                ui.color_edit_button_rgb(&mut light.light.color.as_mut_array());
                            });
                        });
                    }
                    if ui.button("Deselect").clicked() {
                        self.selected = None;
                    }
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Ambient light");
                        ui.color_edit_button_rgb(&mut self.ambient_light.as_mut_array());
                    });
                });

                ui.collapsing("Scene statistics", |ui| {
                    ui.label(format!("{:6} vertices", self.vert_buf.len()));
                    ui.label(format!("{:6} faces", self.index_buf.len()));
                });
            });
    }
}
