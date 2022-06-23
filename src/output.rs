use crate::align_up;
use std::fs::File;
use std::io::{Result, Seek, SeekFrom, Write};
use std::path::Path;

pub trait Output {
    fn current_address(&self) -> u32;
    fn set_address(&mut self, address: u32) -> Result<()>;
    fn align_address(&mut self, align: u32) -> Result<()>;
    fn write_instruction(&mut self, inst: u32, line: usize, source: &str) -> Result<()>;
    fn write_data(&mut self, data: &[u8], line: usize, source: &str) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
}

pub struct BinaryOutput {
    file: File,
    current_address: u32,
}
impl BinaryOutput {
    pub fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: File::create(path)?,
            current_address: 0,
        })
    }
}
impl Output for BinaryOutput {
    fn current_address(&self) -> u32 {
        self.current_address
    }

    fn set_address(&mut self, address: u32) -> Result<()> {
        let file_length = self.file.seek(SeekFrom::End(0))?;
        if (address as u64) > file_length {
            self.file.set_len(address as u64)?;
        }

        self.file.seek(SeekFrom::Start(address as u64))?;
        self.current_address = address;
        Ok(())
    }

    fn align_address(&mut self, align: u32) -> Result<()> {
        let new_address = align_up(self.current_address, align);

        let file_length = self.file.seek(SeekFrom::End(0))?;
        if (new_address as u64) > file_length {
            self.file.set_len(new_address as u64)?;
        }

        self.file.seek(SeekFrom::Start(new_address as u64))?;
        self.current_address = new_address;
        Ok(())
    }

    fn write_instruction(&mut self, inst: u32, _line: usize, _source: &str) -> Result<()> {
        self.file.write_all(&inst.to_le_bytes())?;
        self.current_address += 4;
        Ok(())
    }

    fn write_data(&mut self, data: &[u8], _line: usize, _source: &str) -> Result<()> {
        self.file.write_all(data)?;
        self.current_address += data.len() as u32;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.file.flush()
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
    file: File,
    current_address: u32,
}
impl AnnotatedOutput {
    pub fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            file: File::create(path)?,
            current_address: 0,
        })
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

    fn write_instruction(&mut self, inst: u32, line: usize, source: &str) -> Result<()> {
        writeln!(
            self.file,
            "{:0>8X}: {:0>8X} | {: >4}: {}",
            self.current_address, inst, line, source
        )?;
        self.current_address += 4;
        Ok(())
    }

    fn write_data(&mut self, data: &[u8], line: usize, source: &str) -> Result<()> {
        writeln!(
            self.file,
            "{:0>8X}:{} | {: >4}: {}",
            self.current_address,
            display_data(data),
            line,
            source
        )?;
        self.current_address += data.len() as u32;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.file.flush()
    }
}
