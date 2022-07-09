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

#[cfg(test)]
pub struct TestOutput {
    output: std::io::Cursor<Vec<u8>>,
    current_address: u32,
}
#[cfg(test)]
impl TestOutput {
    #[inline]
    pub fn new() -> Self {
        Self {
            output: std::io::Cursor::new(Vec::new()),
            current_address: 0,
        }
    }

    #[inline]
    pub fn into_inner(self) -> Vec<u8> {
        self.output.into_inner()
    }
}
#[cfg(test)]
impl Output for TestOutput {
    fn current_address(&self) -> u32 {
        self.current_address
    }

    fn set_address(&mut self, address: u32) -> Result<()> {
        let output_length = self.output.get_ref().len();
        if (address as usize) > output_length {
            self.output.get_mut().resize(address as usize, 0);
        }

        self.output.seek(SeekFrom::Start(address as u64))?;
        self.current_address = address;
        Ok(())
    }

    fn align_address(&mut self, align: u32) -> Result<()> {
        let new_address = align_up(self.current_address, align);

        let output_length = self.output.get_ref().len();
        if (new_address as usize) > output_length {
            self.output.get_mut().resize(new_address as usize, 0);
        }

        self.output.seek(SeekFrom::Start(new_address as u64))?;
        self.current_address = new_address;
        Ok(())
    }

    fn write_instruction(&mut self, inst: u32, _line: usize, _source: &str) -> Result<()> {
        self.output.write_all(&inst.to_le_bytes())?;
        self.current_address += 4;
        Ok(())
    }

    fn write_data(&mut self, data: &[u8], _line: usize, _source: &str) -> Result<()> {
        self.output.write_all(data)?;
        self.current_address += data.len() as u32;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.output.flush()
    }
}

pub struct BinaryOutput {
    file: File,
    base_address: u32,
    current_address: u32,
}
impl BinaryOutput {
    pub fn new(path: &Path, base_address: u32) -> Result<Self> {
        Ok(Self {
            file: File::create(path)?,
            base_address,
            current_address: base_address,
        })
    }
}
impl Output for BinaryOutput {
    fn current_address(&self) -> u32 {
        self.current_address
    }

    fn set_address(&mut self, address: u32) -> Result<()> {
        let relative_address = address.saturating_sub(self.base_address) as u64;

        let file_length = self.file.seek(SeekFrom::End(0))?;
        if relative_address > file_length {
            self.file.set_len(relative_address)?;
        }

        self.file.seek(SeekFrom::Start(relative_address))?;
        self.current_address = address;
        Ok(())
    }

    fn align_address(&mut self, align: u32) -> Result<()> {
        let new_address = align_up(self.current_address, align);
        let relative_address = new_address.saturating_sub(self.base_address) as u64;

        let file_length = self.file.seek(SeekFrom::End(0))?;
        if relative_address > file_length {
            self.file.set_len(relative_address)?;
        }

        self.file.seek(SeekFrom::Start(relative_address))?;
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
            self.current_address,
            inst,
            line + 1,
            source
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
            line + 1,
            source
        )?;
        self.current_address += data.len() as u32;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.file.flush()
    }
}
