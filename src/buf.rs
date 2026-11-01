use crate::disk::PAGE_SIZE;

pub struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferError {
    Overflow,
    BufferTooSmall { needed: usize, actual: usize },
}

impl<'a> BufWriter<'a> {
    pub fn new(buf: &'a mut [u8]) -> Result<Self, BufferError> {
        if buf.len() < PAGE_SIZE {
            return Err(BufferError::BufferTooSmall {
                needed: PAGE_SIZE,
                actual: buf.len(),
            });
        }
        Ok(Self { buf, pos: 0 })
    }

    pub fn write_u8(&mut self, v: u8) -> Result<(), BufferError> {
        self.write_bytes(&[v])
    }

    pub fn write_u16_le(&mut self, v: u16) -> Result<(), BufferError> {
        self.write_bytes(&v.to_le_bytes())
    }

    pub fn write_u32_le(&mut self, v: u32) -> Result<(), BufferError> {
        self.write_bytes(&v.to_le_bytes())
    }

    pub fn write_u64_le(&mut self, v: u64) -> Result<(), BufferError> {
        self.write_bytes(&v.to_le_bytes())
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), BufferError> {
        let end = self
            .pos
            .checked_add(bytes.len())
            .ok_or(BufferError::Overflow)?;
        if end > self.buf.len() {
            return Err(BufferError::BufferTooSmall {
                needed: end,
                actual: self.buf.len(),
            });
        }
        self.buf[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
}

pub struct BufReader<'a> {
    buf: &'a [u8],
    pub pos: usize,
}

impl<'a> BufReader<'a> {
    pub fn new(buf: &'a [u8]) -> Result<Self, BufferError> {
        if buf.len() < PAGE_SIZE {
            return Err(BufferError::BufferTooSmall {
                needed: PAGE_SIZE,
                actual: buf.len(),
            });
        }
        Ok(Self { buf, pos: 0 })
    }

    pub fn read_u8(&mut self) -> Result<u8, BufferError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    pub fn read_u16_le(&mut self) -> Result<u16, BufferError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn read_u32_le(&mut self) -> Result<u32, BufferError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64_le(&mut self) -> Result<u64, BufferError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], BufferError> {
        let end = self.pos.checked_add(n).ok_or(BufferError::Overflow)?;
        if end > self.buf.len() {
            return Err(BufferError::BufferTooSmall {
                needed: end,
                actual: self.buf.len(),
            });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
}

/// Reads a little-endian `u16` at `off`. Panics when `off..off + 2` is outside `buf`.
#[inline]
pub fn get_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(buf[off..off + 2].try_into().expect("2-byte slice"))
}

/// Reads a little-endian `u32` at `off`. Panics when `off..off + 4` is outside `buf`.
#[inline]
pub fn get_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().expect("4-byte slice"))
}

/// Reads a little-endian `u64` at `off`. Panics when `off..off + 8` is outside `buf`.
#[inline]
pub fn get_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().expect("8-byte slice"))
}

/// Writes `v` as a little-endian `u16` at `off`. Panics when `off..off + 2` is outside `buf`.
#[inline]
pub fn put_u16_le(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

/// Writes `v` as a little-endian `u32` at `off`. Panics when `off..off + 4` is outside `buf`.
#[inline]
pub fn put_u32_le(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Writes `v` as a little-endian `u64` at `off`. Panics when `off..off + 8` is outside `buf`.
#[inline]
pub fn put_u64_le(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
