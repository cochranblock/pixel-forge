// Unlicense — cochranblock.org
//! EDM preconditioning (Karras et al. 2022, "Elucidating the Design Space
//! of Diffusion-Based Generative Models").
//!
//! Generalizes the constant-σ z-score: instead of one mean/std rescaling,
//! the input/output of the network are scaled by σ-dependent factors so
//! that activations stay near unit variance at *every* noise level. The
//! network always sees a well-conditioned target regardless of how noisy
//! the input is, which is what makes a tiny model competitive with much
//! larger ones at the high-σ regime where ε-pred and clean-pred both
//! struggle.
//!
//! Math is pure scalars — the same coefficients apply to every pixel of
//! every channel. Tensor adapters live at the call sites (candle for the
//! desktop trainer, any-gpu for the wasm runtime).
//!
//!   c_in   = 1 / sqrt(σ² + σ_data²)
//!   c_out  = σ · σ_data / sqrt(σ_data² + σ²)
//!   c_skip = σ_data² / (σ² + σ_data²)
//!   c_noise = 0.25 · log(σ)
//!
//!   D(x; σ) = c_skip · x + c_out · F(c_in · x, c_noise)
//!
//! Where F is the raw network. Loss is MSE between D(x_noisy; σ) and
//! x_clean, weighted by λ(σ) = 1/c_out² to equalize the per-σ loss.

#[derive(Debug, Clone, Copy)]
pub struct EdmCoeffs {
    pub c_in: f32,
    pub c_out: f32,
    pub c_skip: f32,
    pub c_noise: f32,
    /// Loss weight λ(σ) = 1/c_out². Equalizes the initial loss across σ
    /// so MSE on the model's raw output is comparable at every noise
    /// level. With this weight the loss is in the *clean-image* metric,
    /// not the noise-prediction metric.
    pub loss_weight: f32,
}

impl EdmCoeffs {
    /// Compute all four coefficients + loss weight for a given noise
    /// level σ and dataset σ_data. σ must be > 0; callers should pass
    /// at least σ_min (default 0.002 in EDM) — never 0.
    ///
    /// Loss weight uses Min-SNR-γ clamping (Hang et al. 2023) to keep
    /// the weight bounded — `λ(σ) = 1/c_out²` blows up at low σ and
    /// destabilizes Adam. γ=13 is required for σ_data≈0.46: the minimum
    /// unclamped weight is 1/σ_data²≈4.7, so γ=5 is a no-op at all σ levels.
    pub fn at(sigma: f32, sigma_data: f32) -> Self {
        Self::with_gamma(sigma, sigma_data, 13.0)
    }

    /// Same as [`at`] but lets the caller override the Min-SNR-γ cap.
    /// Pass γ = f32::INFINITY for the unclamped EDM weighting.
    pub fn with_gamma(sigma: f32, sigma_data: f32, gamma: f32) -> Self {
        let s2 = sigma * sigma;
        let sd2 = sigma_data * sigma_data;
        let denom = (s2 + sd2).max(1e-20);
        let denom_sqrt = denom.sqrt();
        let c_in = 1.0 / denom_sqrt;
        let c_out = sigma * sigma_data / denom_sqrt;
        let c_skip = sd2 / denom;
        let c_noise = 0.25 * sigma.max(1e-20).ln();
        let raw_weight = 1.0 / (c_out * c_out).max(1e-20);
        let loss_weight = raw_weight.min(gamma);
        Self { c_in, c_out, c_skip, c_noise, loss_weight }
    }
}

/// EDM noise schedule: log-normal during training, ρ-stretched during
/// sampling. These match the published defaults; tweak only with care.
pub const SIGMA_MIN: f32 = 0.002;
pub const SIGMA_MAX: f32 = 80.0;
/// Mean of the training-time log-normal σ distribution (in log space).
pub const P_MEAN: f32 = -1.2;
/// Std of the training-time log-normal σ distribution (in log space).
pub const P_STD: f32 = 1.2;
/// Stretching factor for the deterministic sampler's σ schedule. Lower
/// values pack more steps near σ_min where they matter most.
pub const RHO: f32 = 7.0;

/// Sample a training-time σ from log-normal(P_MEAN, P_STD). Pass a fresh
/// uniform-noise float u ~ U(0,1). Equivalent to:
///   ε ~ N(0,1), σ = exp(P_MEAN + P_STD · ε)
/// but parameterized by u so callers can stay deterministic.
pub fn sigma_from_uniform(u: f32) -> f32 {
    // Inverse CDF of standard normal, via a rational approximation that
    // is accurate to ~1e-7 — fine for picking a σ.
    let eps = inverse_normal_cdf(u.clamp(1e-7, 1.0 - 1e-7));
    (P_MEAN + P_STD * eps).exp()
}

/// Deterministic σ schedule for sampling, EDM style:
///   σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ
/// with σ_N = 0 appended. Returns N+1 values; last one is exactly 0.
pub fn sampling_sigmas(steps: usize) -> Vec<f32> {
    let n = steps as f32;
    let inv_rho = 1.0 / RHO;
    let s_min_p = SIGMA_MIN.powf(inv_rho);
    let s_max_p = SIGMA_MAX.powf(inv_rho);
    let mut out = Vec::with_capacity(steps + 1);
    for i in 0..steps {
        let frac = i as f32 / (n - 1.0).max(1.0);
        let v = s_max_p + frac * (s_min_p - s_max_p);
        out.push(v.powf(RHO));
    }
    out.push(0.0);
    out
}

/// Beasley-Springer-Moro inverse normal CDF approximation. Used to map a
/// uniform [0,1) sample into a standard-normal sample for σ_from_uniform.
fn inverse_normal_cdf(p: f32) -> f32 {
    // Constants from Beasley-Springer (1977), Moro (1995) refinement.
    const A: [f64; 4] = [
         2.50662823884,
       -18.61500062529,
        41.39119773534,
       -25.44106049637,
    ];
    const B: [f64; 4] = [
         -8.47351093090,
         23.08336743743,
        -21.06224101826,
          3.13082909833,
    ];
    const C: [f64; 9] = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];
    let p = p as f64;
    let y = p - 0.5;
    let r = if y.abs() < 0.42 {
        let r = y * y;
        let num = ((A[3]*r + A[2])*r + A[1])*r + A[0];
        let den = (((B[3]*r + B[2])*r + B[1])*r + B[0])*r + 1.0;
        y * num / den
    } else {
        let r = if y > 0.0 { 1.0 - p } else { p };
        let r = (-(r.ln())).ln();
        let mut acc = C[8];
        for i in (0..8).rev() {
            acc = acc * r + C[i];
        }
        if y < 0.0 { -acc } else { acc }
    };
    r as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coeffs_at_zero_sigma_recover_clean_image() {
        // σ → 0 (in practice σ_min): c_skip → 1, c_out → ~σ.
        // Network output drops out; D(x;σ) ≈ x_clean.
        let c = EdmCoeffs::at(SIGMA_MIN, 0.4);
        assert!(c.c_skip > 0.999, "c_skip = {} (expected ~1)", c.c_skip);
        // c_out = σ·σ_data/√(σ²+σ_data²) ≈ σ when σ ≪ σ_data.
        assert!(c.c_out < 1.1 * SIGMA_MIN,
            "c_out = {} (expected ≲ {})", c.c_out, SIGMA_MIN);
    }

    #[test]
    fn coeffs_at_high_sigma_let_network_dominate() {
        // σ ≫ σ_data: c_skip → 0, c_out → σ_data.
        let c = EdmCoeffs::at(SIGMA_MAX, 0.4);
        assert!(c.c_skip < 1e-3, "c_skip = {}", c.c_skip);
        assert!((c.c_out - 0.4).abs() < 1e-3, "c_out = {}", c.c_out);
    }

    #[test]
    fn unclamped_loss_weight_is_inverse_c_out_squared() {
        let c = EdmCoeffs::with_gamma(0.5, 0.4, f32::INFINITY);
        let expected = 1.0 / (c.c_out * c.c_out);
        assert!((c.loss_weight - expected).abs() / expected < 1e-5);
    }

    #[test]
    fn min_snr_gamma_clamps_low_sigma_explosion() {
        // At σ=0.005 σ_data=0.4: c_out ≈ 0.005, raw weight ≈ 40000.
        // Min-SNR-γ=5 should cap it.
        let c = EdmCoeffs::with_gamma(0.005, 0.4, 5.0);
        assert!((c.loss_weight - 5.0).abs() < 1e-5,
            "loss_weight = {} (expected 5)", c.loss_weight);
    }

    #[test]
    fn min_snr_gamma_passthrough_high_sigma() {
        // At σ=10, σ_data=0.4: c_out ≈ 0.4, raw weight ≈ 6.25.
        // Above γ=5 → cap fires and clamps to 5.
        // Below γ=10 → passes through.
        let c10 = EdmCoeffs::with_gamma(10.0, 0.4, 10.0);
        assert!(c10.loss_weight < 10.0,
            "loss_weight = {} (expected < 10)", c10.loss_weight);
    }

    #[test]
    fn sampling_sigmas_endpoints() {
        let s = sampling_sigmas(40);
        assert_eq!(s.len(), 41);
        assert!((s[0] - SIGMA_MAX).abs() < 1e-3);
        assert_eq!(s[40], 0.0);
        // Monotone decreasing
        for w in s.windows(2) {
            assert!(w[1] < w[0], "schedule non-monotone: {:?}", w);
        }
    }

    #[test]
    fn sigma_from_uniform_distributes_across_range() {
        let s_low  = sigma_from_uniform(0.001);
        let s_med  = sigma_from_uniform(0.5);
        let s_high = sigma_from_uniform(0.999);
        // Median is exp(P_MEAN) ≈ 0.30
        assert!((s_med - (P_MEAN.exp())).abs() < 1e-3);
        assert!(s_low < s_med);
        assert!(s_high > s_med);
    }

    #[test]
    fn inverse_normal_cdf_matches_known_quantiles() {
        // Standard normal quantiles to ~3 decimals.
        let pairs = [
            (0.5,    0.0),
            (0.8413, 1.0),
            (0.9772, 2.0),
            (0.1587, -1.0),
        ];
        for (p, want) in pairs {
            let got = inverse_normal_cdf(p);
            assert!((got - want).abs() < 1e-2,
                "Φ⁻¹({p}) = {got}, want {want}");
        }
    }
}
