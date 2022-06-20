use crate::align_up;
use std::io::{Result, Write};

pub trait Output {
    fn current_address(&self) -> u32;
    fn set_address(&mut self, address: u32) -> Result<()>;
    fn align_address(&mut self, align: u32) -> Result<()>;
    fn write_instruction(
        &mut self,
        writer: &mut dyn Write,
        inst: u32,
        line: usize,
        source: &str,
    ) -> Result<()>;
    fn write_data(
        &mut self,
        writer: &mut dyn Write,
        data: &[u8],
        line: usize,
        source: &str,
    ) -> Result<()>;
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

    fn write_instruction(
        &mut self,
        writer: &mut dyn Write,
        inst: u32,
        line: usize,
        source: &str,
    ) -> Result<()> {
        todo!()
    }

    fn write_data(
        &mut self,
        writer: &mut dyn Write,
        data: &[u8],
        line: usize,
        source: &str,
    ) -> Result<()> {
        todo!()
    }
}

fn display_data(data: &[u8]) -> String {
    use std::fmt::Write;

    let mut result = String::with_capacity(data.len() * 3);
    for byte in data.iter().copied() {
        _ = write!(result, " {:0>2X}", byte);
    }
    result
}

pub struct AnnotatedOutput {
    current_address: u32,
}
impl AnnotatedOutput {
    pub fn new() -> Self {
        Self { current_address: 0 }
    }
}
impl Output for AnnotatedOutput {
    fn current_address(&self) -> u32 {
        self.current_address
    }

    fn set_address(&mut self, address: u32) -> Result<()> {
        self.current_address = address;
        Ok(())
    }

    fn align_address(&mut self, align: u32) -> Result<()> {
        self.current_address = align_up(self.current_address, align);
        Ok(())
    }

    fn write_instruction(
        &mut self,
        writer: &mut dyn Write,
        inst: u32,
        line: usize,
        source: &str,
    ) -> Result<()> {
        writeln!(
            writer,
            "{:0>8X}: {:0>8X} | {: >4}: {}",
            self.current_address, inst, line, source
        )?;
        self.current_address += 4;
        Ok(())
    }

    fn write_data(
        &mut self,
        writer: &mut dyn Write,
        data: &[u8],
        line: usize,
        source: &str,
    ) -> Result<()> {
        writeln!(
            writer,
            "{:0>8X}:{} | {: >4}: {}",
            self.current_address,
            display_data(data),
            line,
            source
        )?;
        self.current_address += data.len() as u32;
        Ok(())
    }
}
