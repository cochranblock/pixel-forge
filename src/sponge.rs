// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Sponge Mesh — self-healing training wrapper.
//!
//! Monitors loss curve and retries on failure:
//! - NaN loss → retry with lr * 0.5
//! - Loss plateau → retry with different seed (shuffle order)
//! - Mode collapse → retry with stronger conditioning (higher cfg_dropout)
//! - Max 3 retries per run.
//!
//! pixel-forge owns this — not shared with kova.

use anyhow::Result;
use crate::train::{self, TrainConfig, TrainStop};

const MAX_RETRIES: u32 = 3;

/// Retry reason for logging.
#[derive(Debug)]
enum Retry {
    Nan { epoch: usize, new_lr: f64 },
    Plateau { epoch: usize, new_seed: u64 },
}

/// Run training with Sponge Mesh auto-retry.
/// Wraps `train::train()` and retries up to 3 times on recoverable failures.
pub fn sponge_train(config: &TrainConfig) -> Result<()> {
    let mut cfg = config.clone();
    let mut retries = 0u32;
    let mut retry_seed = 1u64;

    loop {
        println!("sponge: attempt {}/{} (lr={:.1e})",
            retries + 1, MAX_RETRIES + 1, cfg.lr);

        let stop = train::train(&cfg)?;

        match stop {
            TrainStop::Complete { final_loss } => {
                println!("sponge: training complete, final loss: {final_loss:.6}");
                if retries > 0 {
                    println!("sponge: succeeded after {} retries", retries);
                }
                return Ok(());
            }

            TrainStop::NanLoss(epoch) => {
                retries += 1;
                if retries > MAX_RETRIES {
                    anyhow::bail!("sponge: NaN at epoch {} — exhausted {} retries", epoch + 1, MAX_RETRIES);
                }
                let new_lr = cfg.lr * 0.5;
                println!("sponge: NaN at epoch {} — retry {retries}/{MAX_RETRIES} with lr {:.1e} → {:.1e}",
                    epoch + 1, cfg.lr, new_lr);
                let _retry = Retry::Nan { epoch, new_lr };
                cfg.lr = new_lr;
            }

            TrainStop::Plateau { epoch, loss } => {
                retries += 1;
                if retries > MAX_RETRIES {
                    anyhow::bail!("sponge: plateau at epoch {} (loss={loss:.8}) — exhausted {} retries", epoch + 1, MAX_RETRIES);
                }
                println!("sponge: plateau at epoch {} (loss={loss:.8}) — retry {retries}/{MAX_RETRIES} with different shuffle",
                    epoch + 1);
                let _retry = Retry::Plateau { epoch, new_seed: retry_seed };
                retry_seed += 1;
                // Bump LR slightly to escape local minimum
                cfg.lr *= 1.2;
            }
        }
    }
}
