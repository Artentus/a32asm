use std::io::Result;

pub trait Output {
    fn current_address(&self) -> u32;
    fn set_address(&mut self, address: u32) -> Result<()>;
    fn align_address(&mut self, align: u32) -> Result<()>;
    fn write(&mut self, data: &[u8], line: usize, source: &str) -> Result<()>;
}

pub struct BinaryOutput {}
impl BinaryOutput {
    pub fn new() -> Self {
        Self {}
    }
}
impl Output for BinaryOutput {
    fn current_address(&self) -> u32 {
        todo!()
    }

    fn set_address(&mut self, address: u32) -> Result<()> {
        todo!()
    }

    fn align_address(&mut self, align: u32) -> Result<()> {
        todo!()
    }

    fn write(&mut self, data: &[u8], line: usize, source: &str) -> Result<()> {
        todo!()
    }
}
