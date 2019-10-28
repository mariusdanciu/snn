pub struct BinaryOps;

impl BinaryOps {
    pub fn to_u32(b: &[u8]) -> u32 {
        ((b[0] as u32) << 24) +
            ((b[1] as u32) << 16) +
            ((b[2] as u32) << 8) +
            (b[3] as u32)
    }
}
