// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Proof of Artifacts (PoA) — cryptographic signing for generated pixel art.
//!
//! Signs the SHA-256 hash of a 32x32 pixel array + GPS coordinates using Ed25519.
//! Produces a "Signed Heartbeat" packet under 255 bytes for Ghost Fabric LoRa TX.
//!
//! Packet layout (153 bytes):
//!   [0]       version (1 byte)
//!   [1..33]   artifact hash — SHA-256 of 32x32x3 pixel data (32 bytes)
//!   [33..37]  class_id (4 bytes, little-endian u32)
//!   [37..41]  quality_score (4 bytes, f32 LE)
//!   [41..45]  latitude (4 bytes, f32 LE) — Dundalk default: 39.2504
//!   [45..49]  longitude (4 bytes, f32 LE) — Dundalk default: -76.5205
//!   [49..57]  timestamp (8 bytes, u64 LE — unix seconds)
//!   [57..89]  public key (32 bytes, Ed25519)
//!   [89..153] signature (64 bytes, Ed25519)
//!
//! Total: 153 bytes — well under the 255-byte LoRa MTU.

use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Verifier, Signature};
use image::RgbaImage;
use sha2::{Sha256, Digest};

/// PoA protocol version.
const POA_VERSION: u8 = 1;

/// Default GPS: Dundalk, MD (Sollers Point corridor).
pub const DUNDALK_LAT: f32 = 39.2504;
pub const DUNDALK_LON: f32 = -76.5205;

/// Ghost Fabric packet — signed artifact ready for LoRa TX.
/// Exactly 153 bytes. Fits in a single SX1262 frame.
#[derive(Clone, Debug)]
pub struct GhostPacket {
    pub version: u8,
    pub artifact_hash: [u8; 32],
    pub class_id: u32,
    pub quality_score: f32,
    pub latitude: f32,
    pub longitude: f32,
    pub timestamp: u64,
    pub public_key: [u8; 32],
    pub signature: [u8; 64],
}

impl GhostPacket {
    /// Total wire size.
    pub const SIZE: usize = 153;

    /// Serialize to bytes for LoRa transmission.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.version;
        buf[1..33].copy_from_slice(&self.artifact_hash);
        buf[33..37].copy_from_slice(&self.class_id.to_le_bytes());
        buf[37..41].copy_from_slice(&self.quality_score.to_le_bytes());
        buf[41..45].copy_from_slice(&self.latitude.to_le_bytes());
        buf[45..49].copy_from_slice(&self.longitude.to_le_bytes());
        buf[49..57].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[57..89].copy_from_slice(&self.public_key);
        buf[89..153].copy_from_slice(&self.signature);
        buf
    }

    /// Deserialize from LoRa bytes.
    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Self {
        Self {
            version: buf[0],
            artifact_hash: buf[1..33].try_into().unwrap(),
            class_id: u32::from_le_bytes(buf[33..37].try_into().unwrap()),
            quality_score: f32::from_le_bytes(buf[37..41].try_into().unwrap()),
            latitude: f32::from_le_bytes(buf[41..45].try_into().unwrap()),
            longitude: f32::from_le_bytes(buf[45..49].try_into().unwrap()),
            timestamp: u64::from_le_bytes(buf[49..57].try_into().unwrap()),
            public_key: buf[57..89].try_into().unwrap(),
            signature: buf[89..153].try_into().unwrap(),
        }
    }

    /// The signed payload is bytes [0..89] — everything except the signature.
    fn signed_payload(&self) -> [u8; 89] {
        let full = self.to_bytes();
        full[..89].try_into().unwrap()
    }

    /// Verify the packet signature against the embedded public key.
    pub fn verify(&self) -> bool {
        let Ok(vk) = VerifyingKey::from_bytes(&self.public_key) else { return false };
        let sig = Signature::from_bytes(&self.signature);
        vk.verify(&self.signed_payload(), &sig).is_ok()
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "PoA v{} | hash={} | class={} | score={:.3} | gps=({:.4},{:.4}) | ts={} | pk={} | sig={} | {}B",
            self.version,
            hex(&self.artifact_hash[..8]),
            self.class_id,
            self.quality_score,
            self.latitude,
            self.longitude,
            self.timestamp,
            hex(&self.public_key[..8]),
            hex(&self.signature[..8]),
            Self::SIZE,
        )
    }
}

/// Hash a 32x32 RGBA sprite — SHA-256 over the raw RGB channels (3072 bytes).
pub fn hash_sprite(sprite: &RgbaImage) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for y in 0..sprite.height().min(32) {
        for x in 0..sprite.width().min(32) {
            let p = sprite.get_pixel(x, y);
            hasher.update(&[p[0], p[1], p[2]]);
        }
    }
    hasher.finalize().into()
}

/// Hash raw pixel data (channel-first f32 array, 3*32*32 = 3072 floats).
pub fn hash_pixels(pixels: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &v in pixels {
        hasher.update(v.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Generate a new Ed25519 keypair. Returns (signing_key_bytes, verifying_key_bytes).
pub fn generate_keypair() -> ([u8; 32], [u8; 32]) {
    let sk = SigningKey::generate(&mut rand::thread_rng());
    let vk = sk.verifying_key();
    (sk.to_bytes(), vk.to_bytes())
}

/// Load or generate a node keypair from disk.
/// Stored at `~/.pixel-forge/node.key` (32-byte secret).
pub fn load_or_create_keypair() -> anyhow::Result<SigningKey> {
    let dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home dir"))?
        .join(".pixel-forge");
    std::fs::create_dir_all(&dir)?;
    let key_path = dir.join("node.key");

    if key_path.exists() {
        let bytes = std::fs::read(&key_path)?;
        if bytes.len() != 32 {
            anyhow::bail!("corrupt node.key — expected 32 bytes, got {}", bytes.len());
        }
        let arr: [u8; 32] = bytes.try_into().unwrap();
        Ok(SigningKey::from_bytes(&arr))
    } else {
        let sk = SigningKey::generate(&mut rand::thread_rng());
        std::fs::write(&key_path, sk.to_bytes())?;
        // Restrict permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))?;
        }
        println!("generated node keypair: {}", key_path.display());
        println!("  public key: {}", hex(&sk.verifying_key().to_bytes()));
        Ok(sk)
    }
}

/// Sign a sprite and produce a Ghost Fabric packet.
pub fn sign_artifact(
    sprite: &RgbaImage,
    class_id: u32,
    quality_score: f32,
    latitude: f32,
    longitude: f32,
    signing_key: &SigningKey,
) -> GhostPacket {
    let artifact_hash = hash_sprite(sprite);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let public_key = signing_key.verifying_key().to_bytes();

    // Build the payload to sign (first 89 bytes)
    let mut payload = [0u8; 89];
    payload[0] = POA_VERSION;
    payload[1..33].copy_from_slice(&artifact_hash);
    payload[33..37].copy_from_slice(&class_id.to_le_bytes());
    payload[37..41].copy_from_slice(&quality_score.to_le_bytes());
    payload[41..45].copy_from_slice(&latitude.to_le_bytes());
    payload[45..49].copy_from_slice(&longitude.to_le_bytes());
    payload[49..57].copy_from_slice(&timestamp.to_le_bytes());
    payload[57..89].copy_from_slice(&public_key);

    let sig = signing_key.sign(&payload);

    GhostPacket {
        version: POA_VERSION,
        artifact_hash,
        class_id,
        quality_score,
        latitude,
        longitude,
        timestamp,
        public_key,
        signature: sig.to_bytes(),
    }
}

/// Full forge pipeline: generate with quality gate → sign → return packet + image.
pub fn forge_and_sign(
    cond: &crate::class_cond::ClassCond,
    count: u32,
    steps: usize,
    threshold: f32,
    max_attempts: u32,
    disc_path: &str,
    latitude: f32,
    longitude: f32,
) -> anyhow::Result<Vec<(RgbaImage, GhostPacket)>> {
    let device = crate::pipeline::best_device();
    let tier = crate::device_cap::best_available();

    let signing_key = load_or_create_keypair()?;

    let sprites = crate::discriminator::generate_with_gate(
        tier, cond, count, steps, threshold, max_attempts, disc_path, &device,
    )?;

    let mut results = Vec::new();
    for sprite in sprites {
        let score = {
            let (_vm, disc) = crate::discriminator::load(disc_path, &device)?;
            let (_, s) = crate::discriminator::quality_gate(&disc, &sprite, 0.0, &device)?;
            s
        };

        let packet = sign_artifact(&sprite, cond.super_id, score, latitude, longitude, &signing_key);
        assert!(packet.verify(), "self-verification failed");
        results.push((sprite, packet));
    }

    Ok(results)
}

/// Hex-encode a byte slice (lowercase).
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_roundtrip() {
        let (sk_bytes, _) = generate_keypair();
        let sk = SigningKey::from_bytes(&sk_bytes);

        let sprite = RgbaImage::new(16, 16);
        let packet = sign_artifact(&sprite, 0, 0.95, DUNDALK_LAT, DUNDALK_LON, &sk);

        // Size check
        let wire = packet.to_bytes();
        assert_eq!(wire.len(), GhostPacket::SIZE);
        assert!(wire.len() <= 255, "exceeds LoRa MTU");

        // Roundtrip
        let decoded = GhostPacket::from_bytes(&wire);
        assert_eq!(decoded.version, POA_VERSION);
        assert_eq!(decoded.class_id, 0);
        assert_eq!(decoded.artifact_hash, packet.artifact_hash);

        // Signature verification
        assert!(decoded.verify(), "signature failed verification");
    }

    #[test]
    fn tampered_packet_fails() {
        let (sk_bytes, _) = generate_keypair();
        let sk = SigningKey::from_bytes(&sk_bytes);

        let sprite = RgbaImage::new(16, 16);
        let mut packet = sign_artifact(&sprite, 0, 0.95, DUNDALK_LAT, DUNDALK_LON, &sk);

        // Tamper with quality score
        packet.quality_score = 1.0;
        assert!(!packet.verify(), "tampered packet should fail");
    }
}
