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
