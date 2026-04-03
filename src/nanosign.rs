// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! NanoSign — 36-byte model file signing via BLAKE3.
//! Spec: github.com/cochranblock/kova/docs/NANOSIGN.md
//!
//! Append NSIG + BLAKE3 hash on save. Verify on load. Reject tampered files.

use std::path::Path;

const MAGIC: &[u8; 4] = b"NSIG";
const SIG_LEN: usize = 36; // 4 magic + 32 hash

pub enum VerifyResult {
    Verified(blake3::Hash),
    Failed { expected: [u8; 32], actual: blake3::Hash },
    Unsigned,
}

/// Sign a file: append NSIG + BLAKE3 hash of all preceding bytes.
pub fn sign(path: &Path) -> std::io::Result<()> {
    let data = std::fs::read(path)?;
    let hash = blake3::hash(&data);
    let mut f = std::fs::OpenOptions::new().append(true).open(path)?;
    use std::io::Write;
    f.write_all(MAGIC)?;
    f.write_all(hash.as_bytes())?;
    Ok(())
}

/// Verify a signed file. Returns Unsigned if no NSIG marker, Failed if tampered.
pub fn verify(path: &Path) -> std::io::Result<VerifyResult> {
    let data = std::fs::read(path)?;
    if data.len() < SIG_LEN {
        return Ok(VerifyResult::Unsigned);
    }
    let (payload, sig) = data.split_at(data.len() - SIG_LEN);
    if &sig[..4] != MAGIC {
        return Ok(VerifyResult::Unsigned);
    }
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&sig[4..]);
    let actual = blake3::hash(payload);
    if actual.as_bytes() == &expected {
        Ok(VerifyResult::Verified(actual))
    } else {
        Ok(VerifyResult::Failed { expected, actual })
    }
}

/// Strip NanoSign signature from a file (for tools that need raw format).
pub fn strip(path: &Path) -> std::io::Result<()> {
    let data = std::fs::read(path)?;
    if data.len() >= SIG_LEN && &data[data.len() - SIG_LEN..data.len() - 32] == MAGIC {
        std::fs::write(path, &data[..data.len() - SIG_LEN])?;
    }
    Ok(())
}

/// Verify and bail if tampered. Unsigned files pass (backward compat).
pub fn verify_or_bail(path: &str) -> anyhow::Result<()> {
    match verify(Path::new(path))? {
        VerifyResult::Verified(hash) => {
            println!("  nanosign: verified (blake3: {})", &hash.to_hex()[..16]);
        }
        VerifyResult::Unsigned => {
            // Backward compat: unsigned files load fine, just no integrity check
        }
        VerifyResult::Failed { expected, actual } => {
            anyhow::bail!(
                "NanoSign verification FAILED for {path}\n  expected: {}\n  actual:   {}\n  File is tampered or corrupted. Refusing to load.",
                hex::encode(&expected),
                &actual.to_hex()[..64]
            );
        }
    }
    Ok(())
}

/// Sign a file and print the hash.
pub fn sign_and_log(path: &str) -> anyhow::Result<()> {
    sign(Path::new(path))?;
    let data = std::fs::read(path)?;
    let hash = blake3::hash(&data[..data.len() - SIG_LEN]);
    println!("  nanosign: signed (blake3: {})", &hash.to_hex()[..16]);
    Ok(())
}

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
