// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Expert heads for MoE diffusion — tiny specialized networks that correct
//! the base model's bottleneck features at different denoising stages.
//!
//! Shape → Color → Detail → Class. ~50K params each, ~200 KB per expert.
//! First MoE diffusion model under 30 MB.

use anyhow::Result;
use candle_core::{DType, Device, Module, Result as CResult, Tensor};
use candle_nn::{self as nn, VarBuilder, VarMap};

/// Bottleneck channel count from MediumUNet (last entry in CHANNELS).
const BOTTLENECK_CH: usize = 128;

/// A single expert head — 2 conv layers producing a correction tensor.
pub struct ExpertHead {
    conv1: nn::Conv2d,
    gn1: nn::GroupNorm,
    conv2: nn::Conv2d,
    out_channels: usize,
}

impl ExpertHead {
    fn new(out_ch: usize, vb: VarBuilder) -> CResult<Self> {
        let conv1 = nn::conv2d(BOTTLENECK_CH, 64, 3,
            nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c1"))?;
        let gn1 = nn::group_norm(8, 64, 1e-5, vb.pp("gn1"))?;
        let conv2 = nn::conv2d(64, out_ch, 3,
            nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c2"))?;
        Ok(Self { conv1, gn1, conv2, out_channels: out_ch })
    }

    /// Forward: bottleneck features → correction tensor.
    pub fn forward(&self, features: &Tensor) -> CResult<Tensor> {
        let h = self.conv1.forward(features)?;
        let h = self.gn1.forward(&h)?;
        let h = nn::ops::silu(&h)?;
        self.conv2.forward(&h)
    }

    pub fn param_count(varmap: &VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}

/// Shape expert — produces a correction in bottleneck space.
/// Learns silhouette clarity: where is the sprite vs background?
pub fn new_shape_expert(vb: VarBuilder) -> CResult<ExpertHead> {
    ExpertHead::new(BOTTLENECK_CH, vb.pp("shape"))
}

/// Color expert — produces a correction in bottleneck space.
/// Learns palette coherence: which colors belong together?
pub fn new_color_expert(vb: VarBuilder) -> CResult<ExpertHead> {
    ExpertHead::new(BOTTLENECK_CH, vb.pp("color"))
}

/// Detail expert — produces a correction in bottleneck space.
/// Learns texture: shading, highlights, dithering patterns.
pub fn new_detail_expert(vb: VarBuilder) -> CResult<ExpertHead> {
    ExpertHead::new(BOTTLENECK_CH, vb.pp("detail"))
}

/// Class expert — produces a correction in bottleneck space.
/// Learns class identity: is this actually what we asked for?
pub fn new_class_expert(vb: VarBuilder) -> CResult<ExpertHead> {
    ExpertHead::new(BOTTLENECK_CH, vb.pp("class"))
}

/// Which expert to use at this denoising stage.
#[derive(Clone, Copy, Debug)]
pub enum ExpertStage {
    Shape,
    Color,
    Detail,
    Class,
    None,
}

/// Route: which expert fires at this step?
pub fn route(step: usize, total_steps: usize) -> ExpertStage {
    let progress = step as f32 / total_steps as f32;
    if progress < 0.25 {
        ExpertStage::Shape
    } else if progress < 0.50 {
        ExpertStage::Color
    } else if progress < 0.75 {
        ExpertStage::Detail
    } else {
        ExpertStage::Class
    }
}

/// All four experts loaded together.
pub struct ExpertSet {
    pub shape: ExpertHead,
    pub color: ExpertHead,
    pub detail: ExpertHead,
    pub class: ExpertHead,
}

impl ExpertSet {
    pub fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            shape: new_shape_expert(vb.pp("experts"))?,
            color: new_color_expert(vb.pp("experts"))?,
            detail: new_detail_expert(vb.pp("experts"))?,
            class: new_class_expert(vb.pp("experts"))?,
        })
    }

    /// Get the expert for a given stage.
    pub fn get(&self, stage: ExpertStage) -> Option<&ExpertHead> {
        match stage {
            ExpertStage::Shape => Some(&self.shape),
            ExpertStage::Color => Some(&self.color),
            ExpertStage::Detail => Some(&self.detail),
            ExpertStage::Class => Some(&self.class),
            ExpertStage::None => None,
        }
    }

    /// Apply expert correction to bottleneck features.
    pub fn correct(&self, features: &Tensor, stage: ExpertStage) -> CResult<Tensor> {
        match self.get(stage) {
            Some(expert) => {
                let correction = expert.forward(features)?;
                features + &correction
            }
            None => Ok(features.clone()),
        }
    }
}

/// Save all experts to one file (atomic).
pub fn save_experts(varmap: &VarMap, path: &str) -> Result<()> {
    let target = std::path::PathBuf::from(path);
    let tmp = target.with_extension("tmp");
    varmap.save(&tmp)?;
    std::fs::rename(&tmp, &target)?;
    Ok(())
}

/// Load experts from file.
pub fn load_experts(path: &str, device: &Device) -> Result<(VarMap, ExpertSet)> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let experts = ExpertSet::new(vb)?;
    varmap.load(path)?;
    Ok((varmap, experts))
}
