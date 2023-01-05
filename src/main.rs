#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![feature(pattern)]
#![feature(try_trait_v2)]
#![feature(int_roundings)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_slice)]
#![feature(trait_alias)]
#![feature(let_chains)]

#[macro_use]
mod int;
use int::*;

mod iter;

mod lexer;
use lexer::*;

mod parser;
use parser::*;

mod output;
use output::*;

use ahash::AHashMap;
use clap::{AppSettings, ArgEnum, Parser};
use langbox::*;
use std::borrow::Cow;
use std::convert::Infallible;
use std::io::Write;
use std::ops::{ControlFlow, FromResidual, Try};
use std::path::PathBuf;
use std::rc::Rc;
use termcolor::{Color, ColorSpec, WriteColor};

pub type SharedString = Rc<str>;

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
    pub fn pretty_print<W: WriteColor + Write>(
        &self,
        writer: &mut W,
        file_server: &FileServer,
    ) -> std::io::Result<()> {
        writer.reset()?;
        writeln!(writer)?;

        let file_text = file_server
            .get_file(self.span.file_id())
            .expect("invalid file")
            .text();

        let full_span = self.span.join(&self.token_span);
        let full_start_line = full_span.start_pos().line_column(file_server).0 as usize;
        let full_end_line = full_span.end_pos().line_column(file_server).0 as usize;
        let start_line = self.token_span.start_pos().line_column(file_server).0 as usize;
        let end_line = self.token_span.end_pos().line_column(file_server).0 as usize;

        let file_lines = file_text.lines().collect::<Vec<_>>();
        let full_lines = &file_lines[full_start_line..=full_end_line];

        let line_number_width = format!("{}", full_end_line + 1).len();

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
            file_server
                .get_file(self.span.start_pos().file_id())
                .expect("invalid file")
                .path()
                .display(),
            self.span.start_pos().line_column(file_server).0 + 1,
            self.span.start_pos().line_column(file_server).1 + 1,
        )?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        writeln!(writer, "{0:w$} | ", "", w = line_number_width)?;

        for (i, full_line) in full_lines.iter().copied().enumerate() {
            writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
            write!(writer, "{} | ", i + full_start_line + 1)?;

            writer.set_color(ColorSpec::new().set_bold(false).set_fg(Some(Color::White)))?;
            writeln!(writer, "{}", full_line)?;

            if ((i + full_start_line) >= start_line) && ((i + full_start_line) <= end_line) {
                writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
                write!(writer, "{0:w$} | ", "", w = line_number_width)?;

                writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(kind_color)))?;
                if (i + full_start_line) == start_line {
                    let column = self.token_span.start_pos().line_column(file_server).1 as usize;
                    let char_count = full_line.chars().count();
                    write!(writer, "{0:w$}", "", w = column)?;
                    writeln!(writer, "{0:^<w$}", "", w = (char_count - column).max(1))?;
                } else if (i + full_start_line) == end_line {
                    let char_count = full_line.chars().count();
                    writeln!(writer, "{0:^<w$}", "", w = char_count.max(1))?;
                } else {
                    let column = self.token_span.start_pos().line_column(file_server).1 as usize;
                    writeln!(writer, "{0:^<w$}", "", w = column.max(1))?;
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

fn associate_label_addresses(lines: &[Line]) -> Result<AHashMap<String, u32>, Message> {
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
                    let msg = Message {
                        kind: MessageKind::Error,
                        token_span: line.span().clone(),
                        span: line.span().clone(),
                        text: format!("label `{}` has already been defined", name).into(),
                    };

                    return Err(msg);
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

    Ok(map)
}

fn find_constants(lines: &[Line]) -> Result<AHashMap<SharedString, &Expression>, Message> {
    let mut map = AHashMap::new();

    for line in lines.iter() {
        if let LineKind::Define(name, expr) = line.kind() {
            if map.insert(Rc::clone(name), expr).is_some() {
                let msg = Message {
                    kind: MessageKind::Error,
                    token_span: line.span().clone(),
                    span: line.span().clone(),
                    text: format!("constant `{}` has already been defined", name).into(),
                };

                return Err(msg);
            }
        }
    }

    Ok(map)
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

fn evaluate(
    expr: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, &Expression>,
    label_map: &AHashMap<String, u32>,
) -> Result<i64, Message> {
    Ok(match expr.kind() {
        ExpressionKind::IntegerConstant(val) => *val,
        ExpressionKind::CharConstant(c) => (*c as u32) as i64,
        ExpressionKind::Label(name) => {
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
                let msg = Message {
                    kind: MessageKind::Error,
                    token_span: expr.span(),
                    span: expr.span(),
                    text: format!("label `{}` is not defined", name).into(),
                };

                return Err(msg);
            }
        }
        ExpressionKind::Define(name) => {
            if let Some(sub_expr) = constant_map.get(name) {
                evaluate(sub_expr, scope, constant_map, label_map)?
            } else {
                let msg = Message {
                    kind: MessageKind::Error,
                    token_span: expr.span(),
                    span: expr.span(),
                    text: format!("constant `{}` is not defined", name).into(),
                };

                return Err(msg);
            }
        }
        ExpressionKind::UnaryOperator(op, sub_expr) => {
            let sub_val = evaluate(sub_expr, scope, constant_map, label_map)?;
            evaluate_unary_op(op, sub_val)
        }
        ExpressionKind::BinaryOperator(op, lhs_expr, rhs_expr) => {
            let lhs_val = evaluate(lhs_expr, scope, constant_map, label_map)?;
            let rhs_val = evaluate(rhs_expr, scope, constant_map, label_map)?;
            evaluate_binary_op(op, lhs_val, rhs_val)
        }
        ExpressionKind::Parenthesized(expr) => evaluate(expr, scope, constant_map, label_map)?,
    })
}

fn evaluate_folded(
    expr: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<i64, Message> {
    Ok(match expr.kind() {
        ExpressionKind::IntegerConstant(val) => *val,
        ExpressionKind::CharConstant(c) => (*c as u32) as i64,
        ExpressionKind::Label(name) => {
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
                let msg = Message {
                    kind: MessageKind::Error,
                    token_span: expr.span(),
                    span: expr.span(),
                    text: format!("label `{}` is not defined", name).into(),
                };

                return Err(msg);
            }
        }
        ExpressionKind::Define(name) => {
            if let Some(val) = constant_map.get(name) {
                *val
            } else {
                let msg = Message {
                    kind: MessageKind::Error,
                    token_span: expr.span(),
                    span: expr.span(),
                    text: format!("constant `{}` is not defined", name).into(),
                };

                return Err(msg);
            }
        }
        ExpressionKind::UnaryOperator(op, sub_expr) => {
            let sub_val = evaluate_folded(sub_expr, scope, constant_map, label_map)?;
            evaluate_unary_op(op, sub_val)
        }
        ExpressionKind::BinaryOperator(op, lhs_expr, rhs_expr) => {
            let lhs_val = evaluate_folded(lhs_expr, scope, constant_map, label_map)?;
            let rhs_val = evaluate_folded(rhs_expr, scope, constant_map, label_map)?;
            evaluate_binary_op(op, lhs_val, rhs_val)
        }
        ExpressionKind::Parenthesized(expr) => {
            evaluate_folded(expr, scope, constant_map, label_map)?
        }
    })
}

fn fold_constants(
    lines: &[Line],
    label_map: &AHashMap<String, u32>,
) -> Result<AHashMap<SharedString, i64>, Message> {
    let unfolded_map = find_constants(lines)?;

    let mut map = AHashMap::new();
    for (name, expr) in unfolded_map.iter() {
        let val = evaluate(expr, &[], &unfolded_map, label_map)?;
        map.insert(name.clone(), val);
    }

    Ok(map)
}

fn tokenize_file<W: WriteColor + Write>(
    file: FileId,
    file_server: &FileServer,
    writer: &mut W,
) -> std::io::Result<(Vec<Token<TokenKind>>, Vec<usize>, bool)> {
    let mut tokens = Vec::new();
    let mut line_bounds = Vec::new();
    let mut has_error = false;

    let lexer =
        Lexer::<lexer::TokenReader, whitespace_mode::RemoveKeepNewLine>::new(file, file_server);

    for token in lexer {
        let span = token.span;

        match token.kind {
            TokenKind::Error {
                message,
                hint_message,
                dummy,
            } => {
                has_error = true;
                message.pretty_print(writer, file, file_server, MessageKind::Error)?;

                if let Some(hint_message) = hint_message {
                    hint_message.pretty_print(writer, file, file_server, MessageKind::Hint)?;
                }

                if let Some(dummy) = dummy {
                    tokens.push(Token { kind: *dummy, span });
                }
            }
            TokenKind::NewLine | TokenKind::Comment { has_new_line: true } => {
                line_bounds.push(tokens.len());
            }
            kind => tokens.push(Token { kind, span }),
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
        fn $name<W: WriteColor + Write>(
            writer: &mut W,
            file_server: &FileServer,
            vals: &[Expression],
            scope: &[&str],
            constant_map: &AHashMap<SharedString, i64>,
            label_map: &AHashMap<String, u32>,
        ) -> std::io::Result<Vec<u8>> {
            let mut bytes = Vec::with_capacity(vals.len() * std::mem::size_of::<$t>());

            for expr in vals.iter() {
                let val = match evaluate_folded(expr, scope, constant_map, label_map) {
                    Ok(val) => val as $t,
                    Err(msg) => {
                        msg.pretty_print(writer, file_server)?;
                        0
                    }
                };

                bytes.extend_from_slice(&val.to_le_bytes());
            }

            Ok(bytes)
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
}

fn encode_alu_instruction(
    op: AluOp,
    d: Register,
    l: Register,
    r: &AluRhs,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let op_bin = (op as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let l_bin = l.0.into_inner() as u32;

    let (r_bin, grp_bin) = match r {
        AluRhs::Register(r) => (r.0.into_inner() as u32, 0b001),
        AluRhs::Immediate(r) => (
            evaluate_folded(r, scope, constant_map, label_map)? as u32,
            0b010,
        ),
    };

    Ok((r_bin << 17) | (l_bin << 12) | (d_bin << 7) | (op_bin << 3) | grp_bin)
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

fn encode_load_instruction(
    kind: LoadKind,
    d: Register,
    s: Register,
    o: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let s_bin = s.0.into_inner() as u32;
    let o_bin = evaluate_folded(o, scope, constant_map, label_map)? as u32;

    Ok((o_bin << 17) | (s_bin << 12) | (d_bin << 7) | (kind_bin << 3) | 0b011)
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

fn encode_store_instruction(
    kind: StoreKind,
    d: Register,
    o: &Expression,
    s: Register,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let s_bin = s.0.into_inner() as u32;
    let o_bin = evaluate_folded(o, scope, constant_map, label_map)? as u32;

    Ok((o_bin << 17) | (d_bin << 12) | (s_bin << 7) | (kind_bin << 3) | 0b011)
}

fn encode_jump_instruction(
    s: Register,
    o: &Expression,
    indirect: bool,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let s_bin = s.0.into_inner() as u32;
    let o_bin = evaluate_folded(o, scope, constant_map, label_map)? as u32;
    let op_bin = if indirect { 0x1 } else { 0x0 };

    Ok((o_bin << 17) | (s_bin << 12) | (op_bin << 3) | 0b100)
}

fn encode_link_instruction(
    d: Register,
    o: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let d_bin = d.0.into_inner() as u32;
    let o_bin = evaluate_folded(o, scope, constant_map, label_map)? as u32;

    Ok((o_bin << 17) | (d_bin << 7) | (0b0100 << 3) | 0b100)
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum UpperImmediateKind {
    Load  = 0x8,
    AddPc = 0x9,
}

fn encode_upper_immediate_instruction(
    kind: UpperImmediateKind,
    d: Register,
    ui: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let ui_bin = evaluate_folded(ui, scope, constant_map, label_map)? as u32;

    Ok((ui_bin & 0x8000_0000)
        | ((ui_bin & 0x3000) << 17)
        | ((ui_bin & 0x7FFF_C000) >> 2)
        | (d_bin << 7)
        | (kind_bin << 3)
        | 0b100)
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ConditionKind {
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

fn encode_branch_instruction(
    kind: ConditionKind,
    d: &Expression,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
    current_address: u32,
    warnings: &mut Vec<Message>,
) -> Result<u32, Message> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = evaluate_folded(d, scope, constant_map, label_map)?;

    if (d_bin & 0x3) != 0 {
        let msg = Message {
            kind: MessageKind::Warning,
            token_span: d.span(),
            span: d.span(),
            text: "branch is not aligned, actual target address will be truncated".into(),
        };

        warnings.push(msg);
    }

    // The offset is relative to the next instruction, not the current one
    let rel = d_bin - (current_address as i64) - 4;
    if let Some(rel) = i22::new(rel) {
        let rel = (rel.into_inner() as u32) & 0x3F_FFFC;

        Ok(((rel & 0x20_0000) << 10)
            | ((rel & 0x3FFC) << 17)
            | ((rel & 0x1F_C000) >> 2)
            | (kind_bin << 3)
            | 0b101)
    } else {
        let msg = Message {
            kind: MessageKind::Error,
            token_span: d.span(),
            span: d.span(),
            text: "branch is out of range".into(),
        };

        return Err(msg);
    }
}

fn encode_move_instruction(
    kind: ConditionKind,
    d: Register,
    l: Register,
    r: &AluRhs,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
) -> Result<u32, Message> {
    let kind_bin = (kind as u8) as u32;
    let d_bin = d.0.into_inner() as u32;
    let l_bin = l.0.into_inner() as u32;

    let (r_bin, grp_bin) = match r {
        AluRhs::Register(r) => (r.0.into_inner() as u32, 0b110),
        AluRhs::Immediate(r) => (
            evaluate_folded(r, scope, constant_map, label_map)? as u32,
            0b111,
        ),
    };

    Ok((r_bin << 17) | (l_bin << 12) | (d_bin << 7) | (kind_bin << 3) | grp_bin)
}

fn encode_instruction(
    inst: &Instruction,
    line_span: &TextSpan,
    scope: &[&str],
    constant_map: &AHashMap<SharedString, i64>,
    label_map: &AHashMap<String, u32>,
    current_address: u32,
    warnings: &mut Vec<Message>,
) -> Result<u32, Message> {
    if (current_address & 0x3) != 0 {
        let msg = Message {
            kind: MessageKind::Warning,
            token_span: line_span.clone(),
            span: line_span.clone(),
            text: "instruction is not 4-byte aligned, behaviour at execution is undefined".into(),
        };

        warnings.push(msg);
    }

    macro_rules! alu {
        ($op:ident, $d:expr, $l: expr, $r: expr) => {
            encode_alu_instruction(AluOp::$op, *$d, *$l, $r, scope, constant_map, label_map)
        };
    }

    macro_rules! ld {
        ($op:ident, $d:expr, $s: expr, $o: expr) => {
            encode_load_instruction(LoadKind::$op, *$d, *$s, $o, scope, constant_map, label_map)
        };
    }

    macro_rules! st {
        ($op:ident, $d:expr, $o: expr, $s: expr) => {
            encode_store_instruction(StoreKind::$op, *$d, $o, *$s, scope, constant_map, label_map)
        };
    }

    macro_rules! ui {
        ($op:ident, $d:expr, $ui:expr) => {
            encode_upper_immediate_instruction(
                UpperImmediateKind::$op,
                *$d,
                $ui,
                scope,
                constant_map,
                label_map,
            )
        };
    }

    macro_rules! br {
        ($cond:ident, $d:expr) => {
            encode_branch_instruction(
                ConditionKind::$cond,
                $d,
                scope,
                constant_map,
                label_map,
                current_address,
                warnings,
            )
        };
    }

    macro_rules! mv {
        ($cond:ident, $d:expr, $l: expr, $r: expr) => {
            encode_move_instruction(
                ConditionKind::$cond,
                *$d,
                *$l,
                $r,
                scope,
                constant_map,
                label_map,
            )
        };
    }

    match inst {
        Instruction::Nop => Ok((0x0 << 3) | 0b000),
        Instruction::Brk => Ok((0x1 << 3) | 0b000),
        Instruction::Hlt => Ok((0x2 << 3) | 0b000),
        Instruction::Err => Ok((0x3 << 3) | 0b000),
        Instruction::Sys => Ok((0x8 << 3) | 0b000),
        Instruction::ClrK => Ok((0x9 << 3) | 0b000),
        Instruction::Add { d, l, r } => alu!(Add, d, l, r),
        Instruction::AddC { d, l, r } => alu!(AddC, d, l, r),
        Instruction::Sub { d, l, r } => alu!(Sub, d, l, r),
        Instruction::SubB { d, l, r } => alu!(SubB, d, l, r),
        Instruction::And { d, l, r } => alu!(And, d, l, r),
        Instruction::Or { d, l, r } => alu!(Or, d, l, r),
        Instruction::Xor { d, l, r } => alu!(Xor, d, l, r),
        Instruction::Shl { d, l, r } => alu!(Shl, d, l, r),
        Instruction::Lsr { d, l, r } => alu!(Lsr, d, l, r),
        Instruction::Asr { d, l, r } => alu!(Asr, d, l, r),
        Instruction::Mul { d, l, r } => alu!(Mul, d, l, r),
        Instruction::Ld { d, s, o } => ld!(Ld, d, s, o),
        Instruction::Ld8 { d, s, o } => ld!(Ld8, d, s, o),
        Instruction::Ld8s { d, s, o } => ld!(Ld8s, d, s, o),
        Instruction::Ld16 { d, s, o } => ld!(Ld16, d, s, o),
        Instruction::Ld16s { d, s, o } => ld!(Ld16s, d, s, o),
        Instruction::In { d, s, o } => ld!(In, d, s, o),
        Instruction::St { d, o, s } => st!(St, d, o, s),
        Instruction::St8 { d, o, s } => st!(St8, d, o, s),
        Instruction::St16 { d, o, s } => st!(St16, d, o, s),
        Instruction::Out { d, o, s } => st!(Out, d, o, s),
        Instruction::Jmp { s, o, indirect } => {
            encode_jump_instruction(*s, o, *indirect, scope, constant_map, label_map)
        }
        Instruction::Link { d, o } => {
            encode_link_instruction(*d, o, scope, constant_map, label_map)
        }
        Instruction::LdUi { d, ui } => ui!(Load, d, ui),
        Instruction::AddPcUi { d, ui } => ui!(AddPc, d, ui),
        Instruction::BrC { d } => br!(Carry, d),
        Instruction::BrZ { d } => br!(Zero, d),
        Instruction::BrS { d } => br!(Sign, d),
        Instruction::BrO { d } => br!(Overflow, d),
        Instruction::BrNc { d } => br!(NotCarry, d),
        Instruction::BrNz { d } => br!(NotZero, d),
        Instruction::BrNs { d } => br!(NotSign, d),
        Instruction::BrNo { d } => br!(NotOverflow, d),
        Instruction::BrULe { d } => br!(UnsignedLessOrEqual, d),
        Instruction::BrUG { d } => br!(UnsignedGreater, d),
        Instruction::BrSL { d } => br!(SignedLess, d),
        Instruction::BrSGe { d } => br!(SignedGreaterOrEqual, d),
        Instruction::BrSLe { d } => br!(SignedLessOrEqual, d),
        Instruction::BrSG { d } => br!(SignedGreater, d),
        Instruction::Jr { d } => br!(Always, d),
        Instruction::MvC { d, l, r } => mv!(Carry, d, l, r),
        Instruction::MvZ { d, l, r } => mv!(Zero, d, l, r),
        Instruction::MvS { d, l, r } => mv!(Sign, d, l, r),
        Instruction::MvO { d, l, r } => mv!(Overflow, d, l, r),
        Instruction::MvNc { d, l, r } => mv!(NotCarry, d, l, r),
        Instruction::MvNz { d, l, r } => mv!(NotZero, d, l, r),
        Instruction::MvNs { d, l, r } => mv!(NotSign, d, l, r),
        Instruction::MvNo { d, l, r } => mv!(NotOverflow, d, l, r),
        Instruction::MvULe { d, l, r } => mv!(UnsignedLessOrEqual, d, l, r),
        Instruction::MvUG { d, l, r } => mv!(UnsignedGreater, d, l, r),
        Instruction::MvSL { d, l, r } => mv!(SignedLess, d, l, r),
        Instruction::MvSGe { d, l, r } => mv!(SignedGreaterOrEqual, d, l, r),
        Instruction::MvSLe { d, l, r } => mv!(SignedLessOrEqual, d, l, r),
        Instruction::MvSG { d, l, r } => mv!(SignedGreater, d, l, r),
        Instruction::Mov { d, s } => mv!(Always, d, &Register::ZERO, &AluRhs::Register(*s)),
        Instruction::LdI { d, s } => mv!(Always, d, &Register::ZERO, &AluRhs::Immediate(s.clone())),
    }
}

fn parse_file<W: WriteColor + Write>(
    file: FileId,
    file_server: &mut FileServer,
    lines: &mut Vec<Line>,
    line_start: &mut usize,
    writer: &mut W,
) -> std::io::Result<bool> {
    let (tokens, line_bounds, mut has_error) = tokenize_file(file, file_server, writer)?;

    for line_end in line_bounds {
        if line_end > *line_start {
            let line = &tokens[*line_start..line_end];

            let mut is_ws = true;
            for token in line.iter() {
                match &token.kind {
                    TokenKind::Comment { .. } => {}
                    _ => is_ws = false,
                }
            }

            if !is_ws {
                match parse_line(line, file_server) {
                    Ok(line) => {
                        if let LineKind::Directive(AssemblerDirective::Include(path)) = line.kind()
                        {
                            let current_dir = file_server
                                .get_file(file)
                                .unwrap()
                                .path()
                                .parent()
                                .expect("invalid file path");
                            let include_file =
                                file_server.register_file(current_dir.join(path.as_ref()))?;
                            has_error |=
                                parse_file(include_file, file_server, lines, line_start, writer)?;
                        } else {
                            lines.push(line);
                        }
                    }
                    Err(err) => {
                        has_error = true;
                        err.pretty_print(writer, file_server)?;
                    }
                }
            }
        }

        *line_start = line_end;
    }

    Ok(has_error)
}

fn assemble(
    file: FileId,
    file_server: &mut FileServer,
    output: &mut dyn Output,
) -> std::io::Result<()> {
    use termcolor::*;

    let stdout = StandardStream::stdout(ColorChoice::Auto);
    let mut stdout = stdout.lock();

    let mut lines = Vec::new();
    let mut line_start = 0;
    let has_error = parse_file(file, file_server, &mut lines, &mut line_start, &mut stdout)?;

    if !has_error {
        match associate_label_addresses(&lines) {
            Ok(label_map) => match fold_constants(&lines, &label_map) {
                Ok(constant_map) => {
                    let mut scope_stack = Vec::new();
                    let mut warnings = Vec::new();

                    for line in lines.iter() {
                        macro_rules! int {
                            ($to_bytes:expr, $vals:expr) => {{
                                let bytes = $to_bytes(
                                    &mut stdout,
                                    file_server,
                                    $vals,
                                    &scope_stack,
                                    &constant_map,
                                    &label_map,
                                )?;

                                output.write_data(
                                    &bytes,
                                    line.number(),
                                    line.span().text(file_server),
                                )?
                            }};
                        }

                        match line.kind() {
                            LineKind::Label(name) => {
                                let (level, name) = count_leading_dots(name);

                                scope_stack.resize(level, "");
                                scope_stack.push(name);
                            }
                            LineKind::Directive(dir) => match dir {
                                AssemblerDirective::Address(address) => {
                                    output.set_address(*address)?
                                }
                                AssemblerDirective::Align(align) => output.align_address(*align)?,
                                AssemblerDirective::Int8(vals) => int!(to_int8_bytes, vals),
                                AssemblerDirective::Int16(vals) => int!(to_int16_bytes, vals),
                                AssemblerDirective::Int32(vals) => int!(to_int32_bytes, vals),
                                AssemblerDirective::Int64(vals) => int!(to_int64_bytes, vals),
                                AssemblerDirective::Ascii(s) => output.write_data(
                                    &to_ascii_bytes(s, false),
                                    line.number(),
                                    line.span().text(file_server),
                                )?,
                                AssemblerDirective::AsciiZ(s) => output.write_data(
                                    &to_ascii_bytes(s, true),
                                    line.number(),
                                    line.span().text(file_server),
                                )?,
                                AssemblerDirective::Utf8(s) => output.write_data(
                                    s.as_bytes(),
                                    line.number(),
                                    line.span().text(file_server),
                                )?,
                                AssemblerDirective::Utf16(s) => output.write_data(
                                    &to_utf16_bytes(s),
                                    line.number(),
                                    line.span().text(file_server),
                                )?,
                                AssemblerDirective::Unicode(s) => output.write_data(
                                    &to_unicode_bytes(s),
                                    line.number(),
                                    line.span().text(file_server),
                                )?,
                                _ => {}
                            },
                            LineKind::Instruction(inst) => {
                                match encode_instruction(
                                    inst,
                                    line.span(),
                                    &scope_stack,
                                    &constant_map,
                                    &label_map,
                                    output.current_address(),
                                    &mut warnings,
                                ) {
                                    Ok(word) => {
                                        output.write_instruction(
                                            word,
                                            line.number(),
                                            line.span().text(file_server),
                                        )?;
                                    }
                                    Err(msg) => {
                                        msg.pretty_print(&mut stdout, file_server)?;

                                        output.write_instruction(
                                            (0x3 << 3) | 0b000u32,
                                            line.number(),
                                            line.span().text(file_server),
                                        )?;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    for msg in warnings.into_iter() {
                        msg.pretty_print(&mut stdout, file_server)?;
                    }
                }
                Err(msg) => msg.pretty_print(&mut stdout, file_server)?,
            },
            Err(msg) => msg.pretty_print(&mut stdout, file_server)?,
        }
    }

    output.flush()
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ArgEnum)]
enum OutputFormat {
    Binary,
    Annotated,
}

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(global_setting(AppSettings::DeriveDisplayOrder))]
struct Args {
    /// Input file
    #[clap(value_parser, value_name = "INPUT FILE")]
    input: PathBuf,

    /// Output file
    #[clap(short, long, value_parser, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Output format
    #[clap(short, long, arg_enum, value_parser, default_value_t = OutputFormat::Binary)]
    format: OutputFormat,

    /// Base address
    #[clap(short, long, value_parser, value_name = "ADDRESS", default_value_t = 0)]
    base: u32,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let output_path = args
        .output
        .unwrap_or_else(|| args.input.with_extension("bin"));

    let mut file_server = FileServer::new();
    let input_file = file_server.register_file(args.input)?;

    match args.format {
        OutputFormat::Binary => assemble(
            input_file,
            &mut file_server,
            &mut BinaryOutput::new(&output_path, args.base)?,
        ),
        OutputFormat::Annotated => assemble(
            input_file,
            &mut file_server,
            &mut AnnotatedOutput::new(&output_path)?,
        ),
    }
}

#[cfg(test)]
fn to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * std::mem::size_of::<u32>());
    for word in words.iter() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
fn test_assembly(input: &'static str, expected_output: &[u32]) {
    let mut file_server = FileServer::new();
    let input_file = file_server.register_file_memory("<test>", input);

    let mut output = TestOutput::new();
    assemble(input_file, &mut file_server, &mut output).unwrap();

    let output = output.into_inner();
    let expected_output = to_bytes(expected_output);
    assert_eq!(output, expected_output);
}

#[test]
fn assembles_nop_instruction() {
    test_assembly("nop", &[0]);
}

#[test]
fn assembles_alu_reg_instruction() {
    test_assembly(
        "and r1, r2, r3",
        &[0b_0000000000_00011_00010_00001_0100_001],
    );
}

#[test]
fn assembles_alu_imm_instruction() {
    test_assembly("xor r4, r5, 10", &[0b_000000000001010_00101_00100_0110_010]);
}

#[test]
fn assembles_alu_imm_negative_instruction() {
    test_assembly(
        "mul r6, r7, -10",
        &[0b_111111111110110_00111_00110_1010_010],
    );
}

#[test]
fn assembles_alu_imm_overflow_instruction() {
    test_assembly(
        "or r8, r9, 1_131_619_459",
        &[0b_010010010000011_01001_01000_0101_010],
    );
}

#[test]
fn assembles_alu_imm_overflow_negative_instruction() {
    test_assembly(
        "add r10, r11, -1_131_619_459",
        &[0b_101101101111101_01011_01010_0000_010],
    );
}

#[test]
fn assembles_link_instruction() {
    test_assembly("link r12, 8", &[0b_000000000001000_00000_01100_0100_100]);
}

#[test]
fn assembles_ldui_instruction() {
    test_assembly(
        "ldui r13, 1_755_879_355",
        &[0b_0_01_11010001010100010_01101_1000_100],
    );
}

#[test]
fn assembles_branch_instruction() {
    test_assembly(
        "jr target\nnop\ntarget:",
        &[0b_0_000000000001_0000000_00000_1111_101, 0],
    );
}

#[test]
fn assembles_branch_negative_instruction() {
    test_assembly(
        "target:\nnop\njr target",
        &[0, 0b_1_111111111110_1111111_00000_1111_101],
    );
}

#[test]
fn assembles_move_instruction() {
    test_assembly("mov r14, r15", &[0b_0000000000_01111_00000_01110_1111_110]);
}

#[test]
fn assembles_ldi_instruction() {
    test_assembly("ldi r16, 1", &[0b_0000000000_00001_00000_10000_1111_111]);
}

#[test]
fn assembles_trailing_comment() {
    test_assembly("nop // comment", &[0]);
    test_assembly("nop /* comment */", &[0]);
    test_assembly("nop /* comment */ // comment", &[0]);
}
