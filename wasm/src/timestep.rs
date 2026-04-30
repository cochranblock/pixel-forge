// Unlicense — cochranblock.org
//! CPU-side sinusoidal timestep embedding.
//!
//! Mirrors `pixel-forge/src/tiny_unet.rs::timestep_embedding`. The browser
//! evaluates this once per generate() call (or once per CFG step) for a
//! handful of timesteps, so doing it on CPU and uploading the result as a
//! constant is dramatically simpler than adding sin/cos kernels to any-gpu.

/// Build the sinusoidal embedding for a batch of timesteps.
///
/// - `timesteps`: scalar values, one per batch element.
/// - `dim`: even output dimension (TIME_DIM in the model — 64 for Cinder).
///
/// Returns a flat row-major buffer of shape (B, dim). Layout matches the
/// Candle implementation: first half is `cos(t * freq_i)`, second half is
/// `sin(t * freq_i)`, mirroring `Tensor::cat(&[cos, sin], 1)`.
pub fn embed(timesteps: &[f32], dim: usize) -> Vec<f32> {
    assert!(dim % 2 == 0, "timestep dim must be even, got {dim}");
    let half = dim / 2;

    // Same exponential decay as the Candle code:
    //   freq[i] = exp(-i * ln(2) * 2 / half)
    // Computed in f64 for parity with the trained model.
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(i as f64 * std::f64::consts::LN_2 * 2.0 / half as f64).exp()) as f32)
        .collect();

    let mut out = Vec::with_capacity(timesteps.len() * dim);
    for &t in timesteps {
        // cos half
        for &f in &freqs {
            out.push((t * f).cos());
        }
        // sin half
        for &f in &freqs {
            out.push((t * f).sin());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_shape_matches_expected() {
        let out = embed(&[0.0, 0.5, 1.0], 64);
        assert_eq!(out.len(), 3 * 64);
    }

    #[test]
    fn embed_t_zero_is_all_ones_then_zeros() {
        // At t=0: cos(0) = 1 for all freqs; sin(0) = 0 for all freqs.
        let out = embed(&[0.0], 8);
        assert_eq!(&out[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&out[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn embed_dim_must_be_even() {
        // odd dim panics
        let r = std::panic::catch_unwind(|| embed(&[0.5], 7));
        assert!(r.is_err());
    }
}
