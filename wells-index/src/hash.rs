use std::fs;
use std::path::Path;

pub fn blake3_file(path: &Path) -> anyhow::Result<[u8; 32]> {
    let bytes = fs::read(path)?;
    let hash = blake3::hash(&bytes);
    let mut result = [0u8; 32];
    result.copy_from_slice(hash.as_bytes());
    Ok(result)
}

pub fn blake3_bytes(bytes: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(bytes);
    let mut result = [0u8; 32];
    result.copy_from_slice(hash.as_bytes());
    result
}
