#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![feature(pattern)]
#![feature(try_trait_v2)]
#![feature(int_roundings)]

#[macro_use]
mod int;
use int::*;

mod lexer;
use lexer::*;

mod parser;
use parser::*;

mod output;
use output::*;

use ahash::AHashMap;
use std::borrow::Cow;
use std::convert::Infallible;
use std::io::Write;
use std::ops::{ControlFlow, FromResidual, Try};
use std::rc::Rc;
use termcolor::{Color, ColorSpec, WriteColor};

#[derive(Debug)]
pub enum OptionalResult<T, E> {
    Some(T),
    None,
    Err(E),
}
impl<T, E> OptionalResult<T, E> {
    #[inline]
    pub fn is_some(&self) -> bool {
        match self {
            OptionalResult::Some(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        match self {
            OptionalResult::None => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_err(&self) -> bool {
        match self {
            OptionalResult::Err(_) => true,
            _ => false,
        }
    }

    pub fn map<M>(self, f: impl FnOnce(T) -> M) -> OptionalResult<M, E> {
        match self {
            Self::Some(val) => OptionalResult::Some(f(val)),
            Self::None => OptionalResult::None,
            Self::Err(err) => OptionalResult::Err(err),
        }
    }
}
impl<T, E> From<Option<T>> for OptionalResult<T, E> {
    fn from(opt: Option<T>) -> Self {
        if let Some(val) = opt {
            Self::Some(val)
        } else {
            Self::None
        }
    }
}
impl<T, E> From<Result<T, E>> for OptionalResult<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(val) => Self::Some(val),
            Err(err) => Self::Err(err),
        }
    }
}
impl<T, E> From<Option<Result<T, E>>> for OptionalResult<T, E> {
    fn from(opt: Option<Result<T, E>>) -> Self {
        if let Some(result) = opt {
            match result {
                Ok(val) => Self::Some(val),
                Err(err) => Self::Err(err),
            }
        } else {
            Self::None
        }
    }
}
impl<T, E> From<Result<Option<T>, E>> for OptionalResult<T, E> {
    fn from(result: Result<Option<T>, E>) -> Self {
        match result {
            Ok(opt) => {
                if let Some(val) = opt {
                    Self::Some(val)
                } else {
                    Self::None
                }
            }
            Err(err) => Self::Err(err),
        }
    }
}
impl<T, E> Into<Option<Result<T, E>>> for OptionalResult<T, E> {
    fn into(self) -> Option<Result<T, E>> {
        match self {
            OptionalResult::Some(val) => Some(Ok(val)),
            OptionalResult::None => None,
            OptionalResult::Err(err) => Some(Err(err)),
        }
    }
}
impl<T, E> Try for OptionalResult<T, E> {
    type Output = Option<T>;
    type Residual = OptionalResult<Infallible, E>;

    fn from_output(output: Self::Output) -> Self {
        output.into()
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Some(val) => ControlFlow::Continue(Some(val)),
            Self::None => ControlFlow::Continue(None),
            Self::Err(err) => ControlFlow::Break(OptionalResult::Err(err)),
        }
    }
}
impl<T, E> FromResidual for OptionalResult<T, E> {
    fn from_residual(residual: <Self as Try>::Residual) -> Self {
        match residual {
            OptionalResult::Err(err) => Self::Err(err),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageKind {
    Error,
    Hint,
    Warning,
}

#[derive(Debug)]
pub struct Message {
    pub kind: MessageKind,
    pub token_span: TextSpan,
    pub span: TextSpan,
    pub text: Cow<'static, str>,
}
impl Message {
    pub fn pretty_print<W: WriteColor + Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.reset()?;
        writeln!(writer)?;

        let full_span = self.span.combine(&self.token_span);
        let full_lines = full_span.enclosing_lines();
        let lines = self.span.split_into_lines();

        let mut line_number_width = 1;
        for line_span in full_lines.iter() {
            line_number_width = line_number_width.max(format!("{}", line_span.line()).len());
        }

        let kind_color = match self.kind {
            MessageKind::Error => Color::Red,
            MessageKind::Hint => Color::Blue,
            MessageKind::Warning => Color::Yellow,
        };

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(kind_color)))?;
        write!(writer, "{:?}", self.kind)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::White)))?;
        writeln!(writer, ": {}", &self.text)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        write!(writer, "{0:w$}--> ", "", w = line_number_width)?;

        writer.set_color(ColorSpec::new().set_bold(false).set_fg(Some(Color::White)))?;
        writeln!(
            writer,
            "{}:{}:{}",
            self.span.file_path().display(),
            self.span.line(),
            self.span.column(),
        )?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        writeln!(writer, "{0:w$} | ", "", w = line_number_width)?;

        let mut line_iter = lines.iter();
        for line_span in full_lines.iter() {
            writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
            write!(writer, "{} | ", line_span.line())?;

            writer.set_color(ColorSpec::new().set_bold(false).set_fg(Some(Color::White)))?;
            writeln!(writer, "{}", line_span.text())?;

            if line_span.line() >= lines[0].line() {
                if let Some(span) = line_iter.next() {
                    writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
                    write!(writer, "{0:w$} | ", "", w = line_number_width)?;

                    writer
                        .set_color(ColorSpec::new().set_bold(false).set_fg(Some(Color::White)))?;
                    write!(writer, "{0:w$}", "", w = span.column())?;

                    writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(kind_color)))?;
                    writeln!(
                        writer,
                        "{0:^<w$}",
                        "",
                        w = span.text().chars().count().max(1),
                    )?;
                }
            }
        }

        writer.reset()?;
        writeln!(writer)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register(u5);
impl Register {
    pub const ZERO: Self = Self(u5!(0));

    pub const RA: Self = Self(u5!(1));
    pub const SP: Self = Self(u5!(2));

    pub const A0: Self = Self(u5!(3));
    pub const A1: Self = Self(u5!(4));
    pub const A2: Self = Self(u5!(5));
    pub const A3: Self = Self(u5!(6));
    pub const A4: Self = Self(u5!(7));
    pub const A5: Self = Self(u5!(8));
    pub const A6: Self = Self(u5!(9));
    pub const A7: Self = Self(u5!(10));

    pub const T0: Self = Self(u5!(11));
    pub const T1: Self = Self(u5!(12));
    pub const T2: Self = Self(u5!(13));
    pub const T3: Self = Self(u5!(14));
    pub const T4: Self = Self(u5!(15));
    pub const T5: Self = Self(u5!(16));
    pub const T6: Self = Self(u5!(17));
    pub const T7: Self = Self(u5!(18));

    pub const S0: Self = Self(u5!(19));
    pub const S1: Self = Self(u5!(20));
    pub const S2: Self = Self(u5!(21));
    pub const S3: Self = Self(u5!(22));
    pub const S4: Self = Self(u5!(23));
    pub const S5: Self = Self(u5!(24));
    pub const S6: Self = Self(u5!(25));
    pub const S7: Self = Self(u5!(26));
    pub const S8: Self = Self(u5!(27));
    pub const S9: Self = Self(u5!(28));
    pub const S10: Self = Self(u5!(29));
    pub const S11: Self = Self(u5!(30));
    pub const S12: Self = Self(u5!(31));
}
impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.into() {
            0 => write!(f, "zero"),

            1 => write!(f, "ra"),
            2 => write!(f, "sp"),

            3 => write!(f, "a0"),
            4 => write!(f, "a1"),
            5 => write!(f, "a2"),
            6 => write!(f, "a3"),
            7 => write!(f, "a4"),
            8 => write!(f, "a5"),
            9 => write!(f, "a6"),
            10 => write!(f, "a7"),

            11 => write!(f, "t0"),
            12 => write!(f, "t1"),
            13 => write!(f, "t2"),
            14 => write!(f, "t3"),
            15 => write!(f, "t4"),
            16 => write!(f, "t5"),
            17 => write!(f, "t6"),
            18 => write!(f, "t7"),

            19 => write!(f, "s0"),
            20 => write!(f, "s1"),
            21 => write!(f, "s2"),
            22 => write!(f, "s3"),
            23 => write!(f, "s4"),
            24 => write!(f, "s5"),
            25 => write!(f, "s6"),
            26 => write!(f, "s7"),
            27 => write!(f, "s8"),
            28 => write!(f, "s9"),
            29 => write!(f, "s10"),
            30 => write!(f, "s11"),
            31 => write!(f, "s12"),

            _ => unreachable!(),
        }
    }
}

fn align_up(address: u32, align: u32) -> u32 {
    address.div_ceil(align) * align
}

fn count_leading_dots(s: &str) -> (usize, &str) {
    let mut count = 0;
    for (p, c) in s.char_indices() {
        if c == '.' {
            count += 1;
        } else {
            return (count, &s[p..]);
        }
    }

    (count, "")
}

fn associate_label_addresses(lines: &[Line]) -> Option<AHashMap<String, u32>> {
    use std::mem::size_of;

    let mut map = AHashMap::new();
    let mut scope_stack = Vec::new();
    let mut address = 0;

    for line in lines.iter() {
        match line.kind() {
            LineKind::Label(name) => {
                let (level, name) = count_leading_dots(name);

                scope_stack.resize(level, "");
                scope_stack.push(name);

                let mut full_name = String::new();
                for scope in scope_stack[..level].iter() {
                    full_name.push_str(scope);
                    full_name.push_str("::");
                }
                full_name.push_str(name);

                if map.insert(full_name, address).is_some() {
                    // TODO: duplicate label error
                    return None;
                }
            }
            LineKind::Directive(dir) => match dir {
                AssemblerDirective::Address(addr) => address = *addr,
                AssemblerDirective::Align(align) => address = align_up(address, *align),
                AssemblerDirective::Int8(vals) => address += (vals.len() * size_of::<u8>()) as u32,
                AssemblerDirective::Int16(vals) => {
                    address += (vals.len() * size_of::<u16>()) as u32
                }
                AssemblerDirective::Int32(vals) => {
                    address += (vals.len() * size_of::<u32>()) as u32
                }
                AssemblerDirective::Int64(vals) => {
                    address += (vals.len() * size_of::<u64>()) as u32
                }
                AssemblerDirective::Ascii(s) => {
                    address += (s.chars().count() * size_of::<u8>()) as u32
                }
                AssemblerDirective::AsciiZ(s) => {
                    address += ((s.chars().count() + 1) * size_of::<u8>()) as u32
                }
                AssemblerDirective::Utf8(s) => address += s.len() as u32,
                AssemblerDirective::Utf16(s) => {
                    address +=
                        (s.chars().map(|c| c.len_utf16()).sum::<usize>() * size_of::<u16>()) as u32
                }
                AssemblerDirective::Unicode(s) => {
                    address += (s.chars().count() * size_of::<char>()) as u32
                }
                _ => {}
            },
            LineKind::Instruction(_) => address += size_of::<u32>() as u32,
            _ => {}
        }
    }

    Some(map)
}

fn find_constants<'a>(lines: &'a [Line]) -> Option<AHashMap<String, &'a Expression<'a>>> {
    let mut map = AHashMap::new();

    for line in lines.iter() {
        if let LineKind::Define(name, expr) = line.kind() {
            if map.insert(name.to_string(), expr).is_some() {
                // TODO: duplicate constant error
                return None;
            }
        }
    }

    Some(map)
}

macro_rules! bool_as_int {
    ($b:expr) => {
        if $b {
            1
        } else {
            0
        }
    };
}

fn evaluate_unary_op(op: &UnaryOperator, sub_val: i64) -> i64 {
    match op {
        UnaryOperator::Positive => sub_val,
        UnaryOperator::Negative => -sub_val,
        UnaryOperator::Not => !sub_val,
    }
}

fn evaluate_binary_op(op: &BinaryOperator, lhs_val: i64, rhs_val: i64) -> i64 {
    match op {
        BinaryOperator::Add => lhs_val + rhs_val,
        BinaryOperator::Subtract => lhs_val - rhs_val,
        BinaryOperator::Multiply => lhs_val * rhs_val,
        BinaryOperator::Divide => lhs_val / rhs_val,
        BinaryOperator::Remainder => lhs_val % rhs_val,
        BinaryOperator::ShiftLeft => lhs_val << (rhs_val as u64),
        BinaryOperator::ShiftRight => ((lhs_val as u64) >> (rhs_val as u64)) as i64,
        BinaryOperator::ShiftRightArithmetic => lhs_val >> (rhs_val as u64),
        BinaryOperator::And => lhs_val & rhs_val,
        BinaryOperator::Or => lhs_val | rhs_val,
        BinaryOperator::Xor => lhs_val ^ rhs_val,
        BinaryOperator::Equals => bool_as_int!(lhs_val == rhs_val),
        BinaryOperator::NotEquals => bool_as_int!(lhs_val != rhs_val),
        BinaryOperator::LessEqual => bool_as_int!(lhs_val <= rhs_val),
        BinaryOperator::Less => bool_as_int!(lhs_val < rhs_val),
        BinaryOperator::GreaterEqual => bool_as_int!(lhs_val >= rhs_val),
        BinaryOperator::Greater => bool_as_int!(lhs_val > rhs_val),
    }
}

fn evaluate<'a, W: WriteColor + Write>(
    writer: &mut W,
    expr: &'a Expression<'a>,
    scope: &[&'a str],
    constant_map: &AHashMap<String, &'a Expression<'a>>,
    label_map: &AHashMap<String, u32>,
) -> Option<i64> {
    Some(match expr {
        Expression::IntegerConstant(val) => *val,
        Expression::CharConstant(c) => (*c as u32) as i64,
        Expression::Label(name) => {
            let (level, name) = count_leading_dots(name);

            let mut full_name = String::new();
            for scope in scope[..level].iter() {
                full_name.push_str(scope);
                full_name.push_str("::");
            }
            full_name.push_str(name);

            if let Some(addr) = label_map.get(&full_name) {
                *addr as i64
            } else {
                // TODO: label not found error
                return None;
            }
        }
        Expression::Define(name) => {
            if let Some(sub_expr) = constant_map.get(*name) {
                evaluate(writer, sub_expr, scope, constant_map, label_map)?
            } else {
                // TODO: constant not found error
                return None;
            }
        }
        Expression::UnaryOperator(op, sub_expr) => {
            let sub_val = evaluate(writer, sub_expr, scope, constant_map, label_map)?;
            evaluate_unary_op(op, sub_val)
        }
        Expression::BinaryOperator(op, lhs_expr, rhs_expr) => {
            let lhs_val = evaluate(writer, lhs_expr, scope, constant_map, label_map)?;
            let rhs_val = evaluate(writer, rhs_expr, scope, constant_map, label_map)?;
            evaluate_binary_op(op, lhs_val, rhs_val)
        }
        Expression::Parenthesized(expr) => evaluate(writer, expr, scope, constant_map, label_map)?,
    })
}

fn evaluate_folded<'a, W: WriteColor + Write>(
    writer: &mut W,
    expr: &'a Expression<'a>,
    scope: &[&'a str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
) -> Option<i64> {
    Some(match expr {
        Expression::IntegerConstant(val) => *val,
        Expression::CharConstant(c) => (*c as u32) as i64,
        Expression::Label(name) => {
            let (level, name) = count_leading_dots(name);

            let mut full_name = String::new();
            for scope in scope[..level].iter() {
                full_name.push_str(scope);
                full_name.push_str("::");
            }
            full_name.push_str(name);

            if let Some(addr) = label_map.get(&full_name) {
                *addr as i64
            } else {
                // TODO: label not found error
                return None;
            }
        }
        Expression::Define(name) => {
            if let Some(val) = constant_map.get(*name) {
                *val
            } else {
                // TODO: constant not found error
                return None;
            }
        }
        Expression::UnaryOperator(op, sub_expr) => {
            let sub_val = evaluate_folded(writer, sub_expr, scope, constant_map, label_map)?;
            evaluate_unary_op(op, sub_val)
        }
        Expression::BinaryOperator(op, lhs_expr, rhs_expr) => {
            let lhs_val = evaluate_folded(writer, lhs_expr, scope, constant_map, label_map)?;
            let rhs_val = evaluate_folded(writer, rhs_expr, scope, constant_map, label_map)?;
            evaluate_binary_op(op, lhs_val, rhs_val)
        }
        Expression::Parenthesized(expr) => {
            evaluate_folded(writer, expr, scope, constant_map, label_map)?
        }
    })
}

fn fold_constants<W: WriteColor + Write>(
    writer: &mut W,
    lines: &[Line],
    label_map: &AHashMap<String, u32>,
) -> Option<AHashMap<String, i64>> {
    let unfolded_map = find_constants(lines)?;

    let mut map = AHashMap::new();
    for (name, expr) in unfolded_map.iter() {
        let val = evaluate(writer, expr, &[], &unfolded_map, label_map)?;
        map.insert(name.clone(), val);
    }

    Some(map)
}

fn tokenize_file<W: WriteColor + Write>(
    file: &Rc<InputFile>,
    out_writer: &mut W,
) -> std::io::Result<(Vec<Token>, Vec<usize>, bool)> {
    let mut tokens = Vec::new();
    let mut line_bounds = Vec::new();
    let mut has_error = false;

    let lexer = Lexer::new(&file);
    for token in lexer {
        match token {
            Ok(token) => match &token.kind() {
                TokenKind::NewLine | TokenKind::Comment { has_new_line: true } => {
                    if line_bounds.last().copied().unwrap_or(0) < tokens.len() {
                        line_bounds.push(tokens.len());
                    }
                }
                _ => tokens.push(token),
            },
            Err(err) => {
                has_error = true;
                err.pretty_print(out_writer)?;
                tokens.push(err.into_dummy_token());
            }
        }
    }

    if line_bounds.last().copied().unwrap_or(0) < tokens.len() {
        line_bounds.push(tokens.len());
    }

    Ok((tokens, line_bounds, has_error))
}

fn to_ascii_bytes(s: &str, null_terminator: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(s.len() + 1);

    for c in s.chars() {
        if c.is_ascii() {
            bytes.push((c as u32) as u8);
        }
    }

    if null_terminator {
        bytes.push(0);
    }

    bytes
}

fn to_utf16_bytes(s: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(s.len() * 4);

    for c in s.chars() {
        let mut buffer = [0; 2];
        let utf16 = c.encode_utf16(&mut buffer);

        for surrogate in utf16.iter().copied() {
            bytes.extend_from_slice(&surrogate.to_le_bytes());
        }
    }

    bytes
}

fn to_unicode_bytes(s: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(s.len() * 4);

    for c in s.chars() {
        let code = c as u32;
        bytes.extend_from_slice(&code.to_le_bytes());
    }

    bytes
}

macro_rules! def_int_to_bytes {
    ($name:ident, $t:ty) => {
        fn $name<'a, W: WriteColor + Write>(
            writer: &mut W,
            vals: &[Expression<'a>],
            scope: &[&'a str],
            constant_map: &AHashMap<String, i64>,
            label_map: &AHashMap<String, u32>,
        ) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(vals.len() * std::mem::size_of::<$t>());

            for expr in vals.iter() {
                let val = evaluate_folded(writer, expr, scope, constant_map, label_map).unwrap_or(0)
                    as $t;
                bytes.extend_from_slice(&val.to_le_bytes());
            }

            bytes
        }
    };
}

def_int_to_bytes!(to_int8_bytes, u8);
def_int_to_bytes!(to_int16_bytes, u16);
def_int_to_bytes!(to_int32_bytes, u32);
def_int_to_bytes!(to_int64_bytes, u64);

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum AluOp {
    Add    = 0x0,
    AddC   = 0x1,
    Sub    = 0x2,
    SubB   = 0x3,
    And    = 0x4,
    Or     = 0x5,
    Xor    = 0x6,
    Shl    = 0x7,
    Lsr    = 0x8,
    Asr    = 0x9,
    Mul    = 0xA,
    MulHuu = 0xB,
    MulHss = 0xC,
    MulHsu = 0xD,
    CSub   = 0xE,
    Slc    = 0xF,
}

fn encode_alu_instruction<W: WriteColor + Write>(
    writer: &mut W,
    op: AluOp,
    d: Register,
    l: Register,
    r: &AluRhs,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
) -> Option<u32> {
    let op_bin = (op as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let l_bin = l.0.into_inner() as u32;

    let (r_bin, grp_bin) = match r {
        AluRhs::Register(r) => (r.0.into_inner() as u32, 0b001),
        AluRhs::Immediate(r) => (
            evaluate_folded(writer, r, scope, constant_map, label_map)? as u32,
            0b010,
        ),
    };

    Some((r_bin << 17) | (l_bin << 12) | (d_bin << 7) | (op_bin << 3) | grp_bin)
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum LoadKind {
    Ld    = 0b0000,
    Ld8   = 0b0001,
    Ld8s  = 0b0101,
    Ld16  = 0b0010,
    Ld16s = 0b0110,
    In    = 0b0011,
}

fn encode_load_instruction<W: WriteColor + Write>(
    writer: &mut W,
    kind: LoadKind,
    d: Register,
    s: Register,
    o: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
) -> Option<u32> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let s_bin = s.0.into_inner() as u32;
    let o_bin = evaluate_folded(writer, o, scope, constant_map, label_map)? as u32;

    Some((o_bin << 17) | (s_bin << 12) | (d_bin << 7) | (kind_bin << 3) | 0b011)
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum StoreKind {
    St   = 0b1000,
    St8  = 0b1001,
    St16 = 0b1010,
    Out  = 0b1011,
}

fn encode_store_instruction<W: WriteColor + Write>(
    writer: &mut W,
    kind: StoreKind,
    d: Register,
    o: &Expression,
    s: Register,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
) -> Option<u32> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let s_bin = s.0.into_inner() as u32;
    let o_bin = evaluate_folded(writer, o, scope, constant_map, label_map)? as u32;

    Some((o_bin << 17) | (s_bin << 12) | (d_bin << 7) | (kind_bin << 3) | 0b011)
}

fn encode_jump_instruction<W: WriteColor + Write>(
    writer: &mut W,
    s: Register,
    o: &Expression,
    indirect: bool,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
) -> Option<u32> {
    let s_bin = s.0.into_inner() as u32;
    let o_bin = evaluate_folded(writer, o, scope, constant_map, label_map)? as u32;
    let op_bin = if indirect { 0x1 } else { 0x0 };

    Some((o_bin << 17) | (s_bin << 12) | (op_bin << 3) | 0b100)
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum BranchKind {
    Carry                = 0x1,
    Zero                 = 0x2,
    Sign                 = 0x3,
    Overflow             = 0x4,
    NotCarry             = 0x5,
    NotZero              = 0x6,
    NotSign              = 0x7,
    NotOverflow          = 0x8,
    UnsignedLessOrEqual  = 0x9,
    UnsignedGreater      = 0xA,
    SignedLess           = 0xB,
    SignedGreaterOrEqual = 0xC,
    SignedLessOrEqual    = 0xD,
    SignedGreater        = 0xE,
    Always               = 0xF,
}

fn encode_branch_instruction<W: WriteColor + Write>(
    writer: &mut W,
    kind: BranchKind,
    d: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
    current_address: u32,
) -> Option<u32> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = evaluate_folded(writer, d, scope, constant_map, label_map)?;

    if (d_bin & 0x3) != 0 {
        // TODO: warn unaligned branch
    }

    let rel = d_bin - (current_address as i64);
    if let Some(rel) = i22::new(rel) {
        let rel = (rel.into_inner() as u32) & 0x3F_FFFC;
        Some(
            ((rel & 0x20_0000) << 10)
                | ((rel & 0x3FFC) << 17)
                | ((rel & 0x1F_C000) >> 2)
                | (kind_bin << 3)
                | 0b101,
        )
    } else {
        // TODO: error out of range branch
        None
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum UpperImmediateKind {
    Load  = 0x0,
    AddPc = 0x1,
}

fn encode_upper_immediate_instruction<W: WriteColor + Write>(
    writer: &mut W,
    kind: UpperImmediateKind,
    d: Register,
    ui: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
) -> Option<u32> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let ui_bin = evaluate_folded(writer, ui, scope, constant_map, label_map)? as u32;

    Some(
        (ui_bin & 0x8000_0000)
            | ((ui_bin & 0x3000) << 29)
            | ((ui_bin & 0x7FFF_C000) << 12)
            | (d_bin << 7)
            | (kind_bin << 3)
            | 0b110,
    )
}

fn encode_instruction<W: WriteColor + Write>(
    writer: &mut W,
    inst: &Instruction,
    scope: &[&str],
    constant_map: &AHashMap<String, i64>,
    label_map: &AHashMap<String, u32>,
    current_address: u32,
) -> Option<u32> {
    if (current_address & 0x3) != 0 {
        // TODO: warn unaligned instruction
    }

    match inst {
        Instruction::Nop => Some((0x0 << 3) | 0b000),
        Instruction::Brk => Some((0x1 << 3) | 0b000),
        Instruction::Hlt => Some((0x2 << 3) | 0b000),
        Instruction::Err => Some((0x3 << 3) | 0b000),
        Instruction::Add { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Add,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::AddC { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::AddC,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Sub { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Sub,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::SubB { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::SubB,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::And { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::And,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Or { d, l, r } => {
            encode_alu_instruction(writer, AluOp::Or, *d, *l, r, scope, constant_map, label_map)
        }
        Instruction::Xor { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Xor,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Shl { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Shl,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Lsr { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Lsr,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Asr { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Asr,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Mul { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::Mul,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::MulHuu { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::MulHuu,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::MulHss { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::MulHss,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::MulHsu { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::MulHsu,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::CSub { d, l, r } => encode_alu_instruction(
            writer,
            AluOp::CSub,
            *d,
            *l,
            r,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Slc { d, s } => encode_alu_instruction(
            writer,
            AluOp::Slc,
            *d,
            *s,
            &AluRhs::Register(Register::ZERO),
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Ld { d, s, o } => encode_load_instruction(
            writer,
            LoadKind::Ld,
            *d,
            *s,
            o,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Ld8 { d, s, o } => encode_load_instruction(
            writer,
            LoadKind::Ld8,
            *d,
            *s,
            o,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Ld8s { d, s, o } => encode_load_instruction(
            writer,
            LoadKind::Ld8s,
            *d,
            *s,
            o,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Ld16 { d, s, o } => encode_load_instruction(
            writer,
            LoadKind::Ld16,
            *d,
            *s,
            o,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Ld16s { d, s, o } => encode_load_instruction(
            writer,
            LoadKind::Ld16s,
            *d,
            *s,
            o,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::In { d, s, o } => encode_load_instruction(
            writer,
            LoadKind::In,
            *d,
            *s,
            o,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::St { d, o, s } => encode_store_instruction(
            writer,
            StoreKind::St,
            *d,
            o,
            *s,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::St8 { d, o, s } => encode_store_instruction(
            writer,
            StoreKind::St8,
            *d,
            o,
            *s,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::St16 { d, o, s } => encode_store_instruction(
            writer,
            StoreKind::St16,
            *d,
            o,
            *s,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Out { d, o, s } => encode_store_instruction(
            writer,
            StoreKind::Out,
            *d,
            o,
            *s,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Jmp { s, o, indirect } => {
            encode_jump_instruction(writer, *s, o, *indirect, scope, constant_map, label_map)
        }
        Instruction::BrC { d } => encode_branch_instruction(
            writer,
            BranchKind::Carry,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrZ { d } => encode_branch_instruction(
            writer,
            BranchKind::Zero,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrS { d } => encode_branch_instruction(
            writer,
            BranchKind::Sign,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrO { d } => encode_branch_instruction(
            writer,
            BranchKind::Overflow,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrNc { d } => encode_branch_instruction(
            writer,
            BranchKind::NotCarry,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrNz { d } => encode_branch_instruction(
            writer,
            BranchKind::NotZero,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrNs { d } => encode_branch_instruction(
            writer,
            BranchKind::NotSign,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrNo { d } => encode_branch_instruction(
            writer,
            BranchKind::NotOverflow,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrULe { d } => encode_branch_instruction(
            writer,
            BranchKind::UnsignedLessOrEqual,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrUG { d } => encode_branch_instruction(
            writer,
            BranchKind::UnsignedGreater,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrSL { d } => encode_branch_instruction(
            writer,
            BranchKind::SignedLess,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrSGe { d } => encode_branch_instruction(
            writer,
            BranchKind::SignedGreaterOrEqual,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrSLe { d } => encode_branch_instruction(
            writer,
            BranchKind::SignedLessOrEqual,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::BrSG { d } => encode_branch_instruction(
            writer,
            BranchKind::SignedGreater,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::Bra { d } => encode_branch_instruction(
            writer,
            BranchKind::Always,
            d,
            scope,
            constant_map,
            label_map,
            current_address,
        ),
        Instruction::LdUi { d, ui } => encode_upper_immediate_instruction(
            writer,
            UpperImmediateKind::Load,
            *d,
            ui,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::AddPcUi { d, ui } => encode_upper_immediate_instruction(
            writer,
            UpperImmediateKind::AddPc,
            *d,
            ui,
            scope,
            constant_map,
            label_map,
        ),
        Instruction::Sys => Some((0x0 << 3) | 0b111),
        Instruction::ClrK => Some((0x1 << 3) | 0b111),
    }
}

const TEST_FILE: &str = "
$UART_DATA_IN      = 0x004 // read-only
$UART_DATA_OUT     = 0x005 // write-only
$UART_INPUT_COUNT  = 0x006 // read-only
$UART_OUTPUT_COUNT = 0x007 // read-only

serial_read_byte:
    .wait:
        in t0, [$UART_INPUT_COUNT]
        test t0
        br.z .wait

    in a0, [$UART_DATA_IN]
    jmp ra
";

fn main() -> std::io::Result<()> {
    use termcolor::*;

    let stdout = StandardStream::stdout(ColorChoice::Auto);
    let mut stdout = stdout.lock();

    let file = InputFile::new_from_memory("test", TEST_FILE);
    let (tokens, line_bounds, mut has_error) = tokenize_file(&file, &mut stdout)?;

    let mut lines = Vec::new();
    let mut line_start = 0;
    for line_end in line_bounds {
        match parse_line(&tokens[line_start..line_end]) {
            Ok(line) => {
                if let LineKind::Directive(AssemblerDirective::Include(path)) = line.kind() {
                    // TODO: tokenize and parse included file
                } else {
                    lines.push(line);
                }
            }
            Err(err) => {
                has_error = true;
                err.pretty_print(&mut stdout)?;
            }
        }

        line_start = line_end;
    }

    if !has_error {
        if let Some(label_map) = associate_label_addresses(&lines) {
            if let Some(constant_map) = fold_constants(&mut stdout, &lines, &label_map) {
                let mut output = AnnotatedOutput::new();
                let output: &mut dyn Output = &mut output;

                let mut scope_stack = Vec::new();

                for line in lines.iter() {
                    match line.kind() {
                        LineKind::Label(name) => {
                            let (level, name) = count_leading_dots(name);

                            scope_stack.resize(level, "");
                            scope_stack.push(name);
                        }
                        LineKind::Directive(dir) => match dir {
                            AssemblerDirective::Address(address) => output.set_address(*address)?,
                            AssemblerDirective::Align(align) => output.align_address(*align)?,
                            AssemblerDirective::Int8(vals) => {
                                let bytes = to_int8_bytes(
                                    &mut stdout,
                                    vals,
                                    &scope_stack,
                                    &constant_map,
                                    &label_map,
                                );
                                output.write_data(
                                    &mut stdout,
                                    &bytes,
                                    line.number(),
                                    line.span().text(),
                                )?
                            }
                            AssemblerDirective::Int16(vals) => {
                                let bytes = to_int16_bytes(
                                    &mut stdout,
                                    vals,
                                    &scope_stack,
                                    &constant_map,
                                    &label_map,
                                );
                                output.write_data(
                                    &mut stdout,
                                    &bytes,
                                    line.number(),
                                    line.span().text(),
                                )?
                            }
                            AssemblerDirective::Int32(vals) => {
                                let bytes = to_int32_bytes(
                                    &mut stdout,
                                    vals,
                                    &scope_stack,
                                    &constant_map,
                                    &label_map,
                                );
                                output.write_data(
                                    &mut stdout,
                                    &bytes,
                                    line.number(),
                                    line.span().text(),
                                )?
                            }
                            AssemblerDirective::Int64(vals) => {
                                let bytes = to_int64_bytes(
                                    &mut stdout,
                                    vals,
                                    &scope_stack,
                                    &constant_map,
                                    &label_map,
                                );
                                output.write_data(
                                    &mut stdout,
                                    &bytes,
                                    line.number(),
                                    line.span().text(),
                                )?
                            }
                            AssemblerDirective::Ascii(s) => output.write_data(
                                &mut stdout,
                                &to_ascii_bytes(s, false),
                                line.number(),
                                line.span().text(),
                            )?,
                            AssemblerDirective::AsciiZ(s) => output.write_data(
                                &mut stdout,
                                &to_ascii_bytes(s, true),
                                line.number(),
                                line.span().text(),
                            )?,
                            AssemblerDirective::Utf8(s) => output.write_data(
                                &mut stdout,
                                s.as_bytes(),
                                line.number(),
                                line.span().text(),
                            )?,
                            AssemblerDirective::Utf16(s) => output.write_data(
                                &mut stdout,
                                &to_utf16_bytes(s),
                                line.number(),
                                line.span().text(),
                            )?,
                            AssemblerDirective::Unicode(s) => output.write_data(
                                &mut stdout,
                                &to_unicode_bytes(s),
                                line.number(),
                                line.span().text(),
                            )?,
                            _ => {}
                        },
                        LineKind::Instruction(inst) => {
                            if let Some(word) = encode_instruction(
                                &mut stdout,
                                inst,
                                &scope_stack,
                                &constant_map,
                                &label_map,
                                output.current_address(),
                            ) {
                                output.write_instruction(
                                    &mut stdout,
                                    word,
                                    line.number(),
                                    line.span().text(),
                                )?;
                            } else {
                                output.write_instruction(
                                    &mut stdout,
                                    (0x3 << 3) | 0b000u32,
                                    line.number(),
                                    line.span().text(),
                                )?;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(())
}
