use crate::iter::*;
use crate::{Register, SharedString};
use std::borrow::Cow;
use std::io::Write;
use std::iter::Peekable;
use std::str::CharIndices;
use termcolor::{Color, ColorSpec, WriteColor};

#[derive(Debug, Clone)]
pub struct ErrorTokenMessage {
    pub message: Cow<'static, str>,
    pub byte_offset: usize,
    pub byte_len: usize,
}

impl ErrorTokenMessage {
    #[inline]
    fn new<S: Into<Cow<'static, str>>>(message: S, byte_offset: usize, byte_len: usize) -> Self {
        Self {
            message: message.into(),
            byte_offset,
            byte_len,
        }
    }

    fn get_line_columns<'a>(&self, text: &'a str) -> (usize, &'a str, usize, usize) {
        let before = &text[..self.byte_offset];
        let after = &text[(self.byte_offset + self.byte_len)..];

        let line_number = before.chars().filter(|c| *c == '\n').count();
        let line_start = before.rfind('\n').unwrap_or(0);
        let line_end = after.find('\n').unwrap_or(after.len()) + self.byte_offset + self.byte_len;
        let line = &text[line_start..line_end];

        let start_column = text[line_start..self.byte_offset].chars().count();
        let end_column = text[(self.byte_offset + self.byte_len)..line_end]
            .chars()
            .count()
            + start_column;

        (line_number, line, start_column, end_column)
    }

    pub fn pretty_print<W: WriteColor + Write>(
        &self,
        writer: &mut W,
        file: langbox::FileId,
        file_server: &langbox::FileServer,
        kind: crate::MessageKind,
    ) -> std::io::Result<()> {
        writer.reset()?;
        writeln!(writer)?;

        let file_text = file_server.get_file(file).expect("invalid file").text();
        let (line_number, line, start_column, end_column) = self.get_line_columns(file_text);
        let line_number = line_number + 1;
        let line_number_width = format!("{}", line_number).len();

        let kind_color = match kind {
            crate::MessageKind::Error => Color::Red,
            crate::MessageKind::Hint => Color::Blue,
            crate::MessageKind::Warning => Color::Yellow,
        };

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(kind_color)))?;
        write!(writer, "{:?}", kind)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::White)))?;
        writeln!(writer, ": {}", &self.message)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        write!(writer, "{0:w$}--> ", "", w = line_number_width)?;

        writer.set_color(ColorSpec::new().set_bold(false).set_fg(Some(Color::White)))?;
        writeln!(
            writer,
            "{}:{}:{}",
            file_server
                .get_file(file)
                .expect("invalid file")
                .path()
                .display(),
            line_number,
            start_column,
        )?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        writeln!(writer, "{0:w$} | ", "", w = line_number_width)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        write!(writer, "{} | ", line_number)?;

        writer.set_color(ColorSpec::new().set_bold(false).set_fg(Some(Color::White)))?;
        writeln!(writer, "{}", line)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(Color::Cyan)))?;
        write!(writer, "{0:w$} | ", "", w = line_number_width)?;

        writer.set_color(ColorSpec::new().set_bold(true).set_fg(Some(kind_color)))?;
        write!(writer, "{0:w$}", "", w = start_column)?;
        writeln!(
            writer,
            "{0:^<w$}",
            "",
            w = (end_column - start_column).max(1)
        )?;

        writer.reset()?;
        writeln!(writer)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Comma,
    Colon,
    Define,
    Equals,
    NotEquals,
    LessThanEquals,
    GreaterThanEquals,
    LessThan,
    GreaterThan,
    Assign,
    ShiftLeft,
    ShiftRightArithmetic,
    ShiftRight,
    Plus,
    Minus,
    Times,
    Divide,
    Remainder,
    Not,
    And,
    Or,
    Xor,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
}

#[rustfmt::skip]
const OPERATOR_MAP: &[(&str, Operator)] = &[
    (","  , Operator::Comma),
    (":"  , Operator::Colon),
    ("$"  , Operator::Define),
    ("==" , Operator::Equals),
    ("!=" , Operator::NotEquals),
    ("<<" , Operator::ShiftLeft),
    (">>>", Operator::ShiftRightArithmetic),
    (">>" , Operator::ShiftRight),
    ("="  , Operator::Assign),
    ("<=" , Operator::LessThanEquals),
    (">=" , Operator::GreaterThanEquals),
    ("<"  , Operator::LessThan),
    (">"  , Operator::GreaterThan),
    ("+"  , Operator::Plus),
    ("-"  , Operator::Minus),
    ("*"  , Operator::Times),
    ("/"  , Operator::Divide),
    ("%"  , Operator::Remainder),
    ("!"  , Operator::Not),
    ("&"  , Operator::And),
    ("|"  , Operator::Or),
    ("^"  , Operator::Xor),
    ("("  , Operator::OpenParen),
    (")"  , Operator::CloseParen),
    ("["  , Operator::OpenBracket),
    ("]"  , Operator::CloseBracket),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Directive {
    Include,
    Address,
    Align,
    Int8,
    Int16,
    Int32,
    Int64,
    Ascii,
    AsciiZ,
    Utf8,
    Utf16,
    Unicode,
}

#[rustfmt::skip]
const DIRECTIVE_MAP: &[(&str, Directive)] = &[
    ("include", Directive::Include),
    ("address", Directive::Address),
    ("align"  , Directive::Align  ),
    ("d8"     , Directive::Int8   ),
    ("d16"    , Directive::Int16  ),
    ("d32"    , Directive::Int32  ),
    ("d64"    , Directive::Int64  ),
    ("ascii"  , Directive::Ascii  ),
    ("asciiz" , Directive::AsciiZ ),
    ("utf8"   , Directive::Utf8   ),
    ("utf16"  , Directive::Utf16  ),
    ("unicode", Directive::Unicode),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Nop,
    Brk,
    Hlt,
    Err,
    Sys,
    ClrK,

    Add,
    AddC,
    Sub,
    SubB,
    And,
    Or,
    Xor,
    Shl,
    Lsr,
    Asr,
    Mul,
    MulHuu,
    MulHss,
    MulHsu,
    CSub,
    Slc,

    Cmp,
    Bit,
    Test,
    Inc,
    IncC,
    Dec,
    DecB,
    Neg,
    NegB,
    Not,

    Ld,
    Ld8,
    Ld8s,
    Ld16,
    Ld16s,
    In,

    St,
    St8,
    St16,
    Out,

    Jmp,
    Link,
    LdUi,
    AddPcUi,

    BrC,
    BrZ,
    BrS,
    BrO,
    BrNc,
    BrNz,
    BrNs,
    BrNo,
    BrEq,
    BrNeq,
    BrUL,
    BrUGe,
    BrULe,
    BrUG,
    BrSL,
    BrSGe,
    BrSLe,
    BrSG,
    Jr,

    MvC,
    MvZ,
    MvS,
    MvO,
    MvNc,
    MvNz,
    MvNs,
    MvNo,
    MvEq,
    MvNeq,
    MvUL,
    MvUGe,
    MvULe,
    MvUG,
    MvSL,
    MvSGe,
    MvSLe,
    MvSG,
    Mov,
    LdI,
}

#[rustfmt::skip]
const KEYWORD_MAP: &[(&str, Keyword)] = &[
    ("nop"    , Keyword::Nop    ),
    ("brk"    , Keyword::Brk    ),
    ("hlt"    , Keyword::Hlt    ),
    ("err"    , Keyword::Err    ),
    ("sys"    , Keyword::Sys    ),
    ("clrk"   , Keyword::ClrK   ),

    ("add"    , Keyword::Add    ),
    ("addc"   , Keyword::AddC   ),
    ("sub"    , Keyword::Sub    ),
    ("subb"   , Keyword::SubB   ),
    ("and"    , Keyword::And    ),
    ("or"     , Keyword::Or     ),
    ("xor"    , Keyword::Xor    ),
    ("shl"    , Keyword::Shl    ),
    ("lsr"    , Keyword::Lsr    ),
    ("asr"    , Keyword::Asr    ),
    ("mul"    , Keyword::Mul    ),
    ("mulhuu" , Keyword::MulHuu ),
    ("mulhss" , Keyword::MulHss ),
    ("mulhsu" , Keyword::MulHsu ),
    ("csub"   , Keyword::CSub   ),
    ("slc"    , Keyword::Slc    ),

    ("cmp"    , Keyword::Cmp    ),
    ("bit"    , Keyword::Bit    ),
    ("test"   , Keyword::Test   ),
    ("inc"    , Keyword::Inc    ),
    ("incc"   , Keyword::IncC   ),
    ("dec"    , Keyword::Dec    ),
    ("decb"   , Keyword::DecB   ),
    ("neg"    , Keyword::Neg    ),
    ("negb"   , Keyword::NegB   ),
    ("not"    , Keyword::Not    ),

    ("ld"     , Keyword::Ld     ),
    ("ld8"    , Keyword::Ld8    ),
    ("ld8s"   , Keyword::Ld8s   ),
    ("ld16"   , Keyword::Ld16   ),
    ("ld16s"  , Keyword::Ld16s  ),
    ("in"     , Keyword::In     ),

    ("st"     , Keyword::St     ),
    ("st8"    , Keyword::St8    ),
    ("st16"   , Keyword::St16   ),
    ("out"    , Keyword::Out    ),

    ("jmp"    , Keyword::Jmp    ),
    ("link"   , Keyword::Link   ),
    ("ldui"   , Keyword::LdUi   ),
    ("addpcui", Keyword::AddPcUi),

    ("br.c"   , Keyword::BrC    ),
    ("br.z"   , Keyword::BrZ    ),
    ("br.s"   , Keyword::BrS    ),
    ("br.o"   , Keyword::BrO    ),
    ("br.nc"  , Keyword::BrNc   ),
    ("br.nz"  , Keyword::BrNz   ),
    ("br.ns"  , Keyword::BrNs   ),
    ("br.no"  , Keyword::BrNo   ),
    ("br.eq"  , Keyword::BrEq   ),
    ("br.neq" , Keyword::BrNeq  ),
    ("br.u.l" , Keyword::BrUL   ),
    ("br.u.ge", Keyword::BrUGe  ),
    ("br.u.le", Keyword::BrULe  ),
    ("br.u.g" , Keyword::BrUG   ),
    ("br.s.l" , Keyword::BrSL   ),
    ("br.s.ge", Keyword::BrSGe  ),
    ("br.s.le", Keyword::BrSLe  ),
    ("br.s.g" , Keyword::BrSG   ),
    ("jr"     , Keyword::Jr     ),

    ("mv.c"   , Keyword::MvC    ),
    ("mv.z"   , Keyword::MvZ    ),
    ("mv.s"   , Keyword::MvS    ),
    ("mv.o"   , Keyword::MvO    ),
    ("mv.nc"  , Keyword::MvNc   ),
    ("mv.nz"  , Keyword::MvNz   ),
    ("mv.ns"  , Keyword::MvNs   ),
    ("mv.no"  , Keyword::MvNo   ),
    ("mv.eq"  , Keyword::MvEq   ),
    ("mv.neq" , Keyword::MvNeq  ),
    ("mv.u.l" , Keyword::MvUL   ),
    ("mv.u.ge", Keyword::MvUGe  ),
    ("mv.u.le", Keyword::MvULe  ),
    ("mv.u.g" , Keyword::MvUG   ),
    ("mv.s.l" , Keyword::MvSL   ),
    ("mv.s.ge", Keyword::MvSGe  ),
    ("mv.s.le", Keyword::MvSLe  ),
    ("mv.s.g" , Keyword::MvSG   ),
    ("mov"    , Keyword::Mov    ),
    ("ldi"    , Keyword::LdI    ),
];

#[rustfmt::skip]
const REGISTER_MAP: &[(&str, Register)] = &[
    ("r0" , Register(u5!(0 ))),
    ("r1" , Register(u5!(1 ))),
    ("r2" , Register(u5!(2 ))),
    ("r3" , Register(u5!(3 ))),
    ("r4" , Register(u5!(4 ))),
    ("r5" , Register(u5!(5 ))),
    ("r6" , Register(u5!(6 ))),
    ("r7" , Register(u5!(7 ))),
    ("r8" , Register(u5!(8 ))),
    ("r9" , Register(u5!(9 ))),
    ("r10", Register(u5!(10))),
    ("r11", Register(u5!(11))),
    ("r12", Register(u5!(12))),
    ("r13", Register(u5!(13))),
    ("r14", Register(u5!(14))),
    ("r15", Register(u5!(15))),
    ("r16", Register(u5!(16))),
    ("r17", Register(u5!(17))),
    ("r18", Register(u5!(18))),
    ("r19", Register(u5!(19))),
    ("r20", Register(u5!(20))),
    ("r21", Register(u5!(21))),
    ("r22", Register(u5!(22))),
    ("r23", Register(u5!(23))),
    ("r24", Register(u5!(24))),
    ("r25", Register(u5!(25))),
    ("r26", Register(u5!(26))),
    ("r27", Register(u5!(27))),
    ("r28", Register(u5!(28))),
    ("r29", Register(u5!(29))),
    ("r30", Register(u5!(30))),
    ("r31", Register(u5!(31))),

    ("zero", Register::ZERO),

    ("ra", Register::RA),
    ("sp", Register::SP),

    ("a0", Register::A0),
    ("a1", Register::A1),
    ("a2", Register::A2),
    ("a3", Register::A3),
    ("a4", Register::A4),
    ("a5", Register::A5),
    ("a6", Register::A6),
    ("a7", Register::A7),

    ("t0", Register::T0),
    ("t1", Register::T1),
    ("t2", Register::T2),
    ("t3", Register::T3),
    ("t4", Register::T4),
    ("t5", Register::T5),
    ("t6", Register::T6),
    ("t7", Register::T7),

    ("s0" , Register::S0 ),
    ("s1" , Register::S1 ),
    ("s2" , Register::S2 ),
    ("s3" , Register::S3 ),
    ("s4" , Register::S4 ),
    ("s5" , Register::S5 ),
    ("s6" , Register::S6 ),
    ("s7" , Register::S7 ),
    ("s8" , Register::S8 ),
    ("s9" , Register::S9 ),
    ("s10", Register::S10),
    ("s11", Register::S11),
    ("s12", Register::S12),
];

#[derive(Debug, Clone)]
pub enum TokenKind {
    Error {
        message: ErrorTokenMessage,
        hint_message: Option<ErrorTokenMessage>,
        dummy: Option<Box<TokenKind>>,
    },
    NewLine,
    Comment {
        has_new_line: bool,
    },
    Operator(Operator),
    IntegerLiteral(i64),
    CharLiteral(char),
    StringLiteral(SharedString),
    Directive(Directive),
    Register(Register),
    Keyword(Keyword),
    Identifier(SharedString),
}

impl TokenKind {
    #[inline]
    fn dummy_integer() -> Box<Self> {
        Box::new(Self::IntegerLiteral(0))
    }

    #[inline]
    fn dummy_char() -> Box<Self> {
        Box::new(Self::CharLiteral('\0'))
    }

    #[inline]
    fn dummy_string() -> Box<Self> {
        Box::new(Self::StringLiteral("".into()))
    }

    #[inline]
    fn dummy_identifier() -> Box<Self> {
        Box::new(Self::Identifier("".into()))
    }
}

impl PartialEq for TokenKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Error { .. }, _) => false,
            (_, Self::Error { .. }) => false,
            (
                Self::Comment {
                    has_new_line: l_has_new_line,
                },
                Self::Comment {
                    has_new_line: r_has_new_line,
                },
            ) => l_has_new_line == r_has_new_line,
            (Self::Operator(l0), Self::Operator(r0)) => l0 == r0,
            (Self::IntegerLiteral(l0), Self::IntegerLiteral(r0)) => l0 == r0,
            (Self::CharLiteral(l0), Self::CharLiteral(r0)) => l0 == r0,
            (Self::StringLiteral(l0), Self::StringLiteral(r0)) => l0 == r0,
            (Self::Directive(l0), Self::Directive(r0)) => l0 == r0,
            (Self::Register(l0), Self::Register(r0)) => l0 == r0,
            (Self::Keyword(l0), Self::Keyword(r0)) => l0 == r0,
            (Self::Identifier(l0), Self::Identifier(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

fn parse_integer(s: &str) -> TokenKind {
    let mut iter = s.char_indices().peekable();
    let (_, first) = iter.next().expect("s should contain at least one char");

    let mut result_str = String::new();
    let radix;
    if first == '0' {
        if let Some((_, prefix)) = iter.peek() {
            result_str.reserve_exact(s.len());

            match prefix {
                'b' | 'B' => {
                    iter.next();
                    radix = 2;
                }
                'o' | 'O' => {
                    iter.next();
                    radix = 8;
                }
                'x' | 'X' => {
                    iter.next();
                    radix = 16;
                }
                _ => {
                    result_str.push(first);
                    radix = 10;
                }
            }
        } else {
            return TokenKind::IntegerLiteral(0);
        }
    } else {
        result_str.reserve_exact(s.len());
        result_str.push(first);
        radix = 10;
    }

    for (p, c) in iter {
        if c == '_' {
            // do nothing
        } else if c.is_digit(radix) {
            result_str.push(c);
        } else {
            return TokenKind::Error {
                message: ErrorTokenMessage::new("illegal character in literal", 0, s.len()),
                hint_message: Some(ErrorTokenMessage::new(
                    "this character is not valid in the literal",
                    p,
                    c.len_utf8(),
                )),
                dummy: Some(TokenKind::dummy_integer()),
            };
        }
    }

    TokenKind::IntegerLiteral(
        i64::from_str_radix(&result_str, radix)
            .expect("result_str should only contain valid digits"),
    )
}

fn parse_hex_escape<const N: usize>(
    iter: &mut Peekable<CharIndices>,
    esc_pos: usize,
    hex_pos: usize,
    kind_string: &str,
    count_string: &str,
) -> Result<char, (ErrorTokenMessage, Option<ErrorTokenMessage>)>
where
    [(); N * 4]: Sized,
{
    let digits = iter.map(|(_, c)| c).next_n::<N>();
    let digit_count = digits.len();
    let digit_str: ValueString<{ N * 4 }> = digits.into();

    if digit_count == N {
        if let Ok(code) = u32::from_str_radix(&digit_str, 16) {
            char::from_u32(code).ok_or_else(|| {
                let msg = ErrorTokenMessage::new(
                    "invalid escape sequence",
                    esc_pos,
                    hex_pos + digit_str.len(),
                );
                let hint_msg = ErrorTokenMessage::new(
                    format!("`{:X}` is not a valid codepoint", code),
                    hex_pos,
                    hex_pos + digit_str.len(),
                );

                (msg, Some(hint_msg))
            })
        } else {
            let msg = ErrorTokenMessage::new(
                "invalid escape sequence",
                esc_pos,
                hex_pos + digit_str.len(),
            );
            let hint_msg = ErrorTokenMessage::new(
                "some characters are not valid hexadecimal digits",
                hex_pos,
                hex_pos + digit_str.len(),
            );

            Err((msg, Some(hint_msg)))
        }
    } else {
        let msg = ErrorTokenMessage::new(
            "incomplete escape sequence",
            esc_pos,
            hex_pos + digit_str.len(),
        );
        let hint_msg = ErrorTokenMessage::new(
            format!(
                "{} escape sequence must contain exactly {} hex digits",
                kind_string, count_string
            ),
            hex_pos,
            hex_pos + digit_str.len(),
        );

        Err((msg, Some(hint_msg)))
    }
}

fn unescape_string(s: &str) -> Result<String, (ErrorTokenMessage, Option<ErrorTokenMessage>)> {
    let mut iter = s.char_indices().peekable();
    let mut result_str = String::with_capacity(s.len());

    while let Some((p, c)) = iter.next() {
        if c == '\\' {
            let (next_p, next_c) = iter.next().expect("escape sequence invariant broken");

            match next_c {
                '0' => {
                    result_str.push('\0');
                }
                'n' => {
                    result_str.push('\n');
                }
                'r' => {
                    result_str.push('\r');
                }
                't' => {
                    result_str.push('\t');
                }
                '\\' => {
                    result_str.push('\\');
                }
                '\'' => {
                    result_str.push('\'');
                }
                '\"' => {
                    result_str.push('\"');
                }
                'x' => {
                    let c = parse_hex_escape::<2>(
                        &mut iter,
                        p,
                        next_p + 'x'.len_utf8(),
                        "an ascii",
                        "two",
                    )?;

                    result_str.push(c);
                }
                'u' => {
                    let c = parse_hex_escape::<4>(
                        &mut iter,
                        p,
                        next_p + 'u'.len_utf8(),
                        "a unicode",
                        "four",
                    )?;

                    result_str.push(c);
                }
                _ => {
                    let msg = ErrorTokenMessage::new(
                        "invalid escape sequence",
                        p,
                        c.len_utf8() + next_c.len_utf8(),
                    );

                    let hint_msg = ErrorTokenMessage::new(
                        "valid escape sequences are `\\0`, `\\n`, `\\r`, `\\t`, `\\\\`, `\\'`, `\\\"`, `\\xXX` and `\\uUUUU`",
                        next_p,
                        next_c.len_utf8()
                    );

                    return Err((msg, Some(hint_msg)));
                }
            }
        } else {
            result_str.push(c);
        }
    }

    Ok(result_str)
}

pub type ReadTokenResult = langbox::ReadTokenResult<TokenKind>;

pub enum TokenReader {}

impl TokenReader {
    fn count_while(text: &str, mut predicate: impl FnMut(char) -> bool) -> usize {
        let mut count = text.len();
        for (p, c) in text.char_indices() {
            if !predicate(c) {
                count = p;
                break;
            }
        }
        count
    }

    fn count_while_escaped(text: &str, predicate: impl Fn(char) -> bool) -> usize {
        let mut prev = None;
        Self::count_while(text, |c| {
            if let Some('\\') = prev {
                prev = None;
                true
            } else {
                prev = Some(c);
                predicate(c)
            }
        })
    }

    fn read_new_line(text: &str) -> Option<ReadTokenResult> {
        if let Some('\n') = text.chars().next() {
            Some(ReadTokenResult {
                token: TokenKind::NewLine,
                consumed_bytes: '\n'.len_utf8(),
            })
        } else {
            None
        }
    }

    fn read_line_comment(text: &str) -> Option<ReadTokenResult> {
        if text.starts_with("//") {
            Some(ReadTokenResult {
                token: TokenKind::Comment {
                    has_new_line: false,
                },
                consumed_bytes: Self::count_while(text, |c| c != '\n'),
            })
        } else {
            None
        }
    }

    fn read_block_comment(text: &str) -> Option<ReadTokenResult> {
        if text.starts_with("/*") {
            let mut pos = "/*".len();

            loop {
                pos += Self::count_while(&text[pos..], |c| c != '*');
                pos += Self::count_while(&text[pos..], |c| c == '*');

                if let Some(c) = text[pos..].chars().next() {
                    if c == '/' {
                        pos += '/'.len_utf8();
                        break;
                    }
                } else {
                    let dummy = TokenKind::Comment {
                        has_new_line: text[..pos].contains('\n'),
                    };

                    return Some(ReadTokenResult {
                        token: TokenKind::Error {
                            message: ErrorTokenMessage::new("open block comment", pos, 0),
                            hint_message: Some(ErrorTokenMessage::new(
                                "block comments need to be closed with `*/`",
                                0,
                                pos,
                            )),
                            dummy: Some(Box::new(dummy)),
                        },
                        consumed_bytes: pos,
                    });
                }
            }

            Some(ReadTokenResult {
                token: TokenKind::Comment {
                    has_new_line: text[..pos].contains('\n'),
                },
                consumed_bytes: pos,
            })
        } else {
            None
        }
    }

    fn read_operator(text: &str) -> Option<ReadTokenResult> {
        for (op_str, op) in OPERATOR_MAP.iter().copied() {
            if text.starts_with(&op_str) {
                return Some(ReadTokenResult {
                    token: TokenKind::Operator(op),
                    consumed_bytes: op_str.len(),
                });
            }
        }

        None
    }

    fn read_integer_literal(text: &str) -> Option<ReadTokenResult> {
        if let Some(first) = text.chars().next() {
            if first.is_ascii_digit() {
                let end_pos = text
                    .find(|c: char| !c.is_alphanumeric() & (c != '_'))
                    .unwrap_or(text.len());

                Some(ReadTokenResult {
                    token: parse_integer(&text[..end_pos]),
                    consumed_bytes: end_pos,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn read_char_literal(text: &str) -> Option<ReadTokenResult> {
        if let Some('\'') = text.chars().next() {
            let start_pos = '\''.len_utf8();
            let end_pos = start_pos + Self::count_while_escaped(&text[start_pos..], |c| c != '\'');

            if let Some('\'') = text[end_pos..].chars().next() {
                let byte_len = end_pos + '\''.len_utf8();

                let token = match unescape_string(&text[start_pos..end_pos]) {
                    Ok(s) => {
                        let mut chars = s.chars();
                        if let Some(c) = chars.next() {
                            if chars.next().is_some() {
                                TokenKind::Error {
                                    message: ErrorTokenMessage::new(
                                        "char literal contains more than one codepoint",
                                        0,
                                        byte_len,
                                    ),
                                    hint_message: None,
                                    dummy: Some(TokenKind::dummy_char()),
                                }
                            } else {
                                TokenKind::CharLiteral(c)
                            }
                        } else {
                            TokenKind::Error {
                                message: ErrorTokenMessage::new("empty char literal", 0, byte_len),
                                hint_message: None,
                                dummy: Some(TokenKind::dummy_char()),
                            }
                        }
                    }
                    Err((msg, hint_msg)) => TokenKind::Error {
                        message: msg,
                        hint_message: hint_msg,
                        dummy: Some(TokenKind::dummy_char()),
                    },
                };

                Some(ReadTokenResult {
                    token,
                    consumed_bytes: byte_len,
                })
            } else {
                return Some(ReadTokenResult {
                    token: TokenKind::Error {
                        message: ErrorTokenMessage::new("open char literal", end_pos, 0),
                        hint_message: Some(ErrorTokenMessage::new(
                            "char literals need to be closed with `'`",
                            0,
                            end_pos,
                        )),
                        dummy: Some(TokenKind::dummy_char()),
                    },
                    consumed_bytes: end_pos,
                });
            }
        } else {
            None
        }
    }

    fn read_string_literal(text: &str) -> Option<ReadTokenResult> {
        if let Some('"') = text.chars().next() {
            let start_pos = '"'.len_utf8();
            let end_pos = start_pos + Self::count_while_escaped(&text[start_pos..], |c| c != '"');

            if let Some('"') = text[end_pos..].chars().next() {
                let token = match unescape_string(&text[start_pos..end_pos]) {
                    Ok(s) => TokenKind::StringLiteral(s.into()),
                    Err((msg, hint_msg)) => TokenKind::Error {
                        message: msg,
                        hint_message: hint_msg,
                        dummy: Some(TokenKind::dummy_char()),
                    },
                };

                Some(ReadTokenResult {
                    token,
                    consumed_bytes: end_pos + '"'.len_utf8(),
                })
            } else {
                return Some(ReadTokenResult {
                    token: TokenKind::Error {
                        message: ErrorTokenMessage::new("open string literal", end_pos, 0),
                        hint_message: Some(ErrorTokenMessage::new(
                            "string literals need to be closed with `\"`",
                            0,
                            end_pos,
                        )),
                        dummy: Some(TokenKind::dummy_string()),
                    },
                    consumed_bytes: end_pos,
                });
            }
        } else {
            None
        }
    }

    fn read_directive(text: &str) -> Option<ReadTokenResult> {
        if let Some('#') = text.chars().next() {
            let start_pos = '#'.len_utf8();
            let byte_len = start_pos
                + Self::count_while(&text[start_pos..], |c| {
                    c.is_alphanumeric() | (c == '_') | (c == '.')
                });
            let ident = &text[start_pos..byte_len];

            if ident.len() == 0 {
                Some(ReadTokenResult {
                    token: TokenKind::Error {
                        message: ErrorTokenMessage::new("expected directive", start_pos, 0),
                        hint_message: Some(ErrorTokenMessage::new(
                            "`#` indicates the start of a directive",
                            0,
                            '#'.len_utf8(),
                        )),
                        dummy: Some(TokenKind::dummy_identifier()),
                    },
                    consumed_bytes: byte_len,
                })
            } else {
                for (d_str, d) in DIRECTIVE_MAP.iter().copied() {
                    if ident.eq_ignore_ascii_case(d_str) {
                        return Some(ReadTokenResult {
                            token: TokenKind::Directive(d),
                            consumed_bytes: byte_len,
                        });
                    }
                }

                Some(ReadTokenResult {
                    token: TokenKind::Error {
                        message: ErrorTokenMessage::new("unknown directive", 0, byte_len),
                        hint_message: None,
                        dummy: None,
                    },
                    consumed_bytes: byte_len,
                })
            }
        } else {
            None
        }
    }

    fn read_identifier(text: &str) -> Option<ReadTokenResult> {
        if let Some(first) = text.chars().next() {
            if first.is_alphabetic() | (first == '_') | (first == '.') {
                let byte_len =
                    Self::count_while(text, |c| c.is_alphanumeric() | (c == '_') | (c == '.'));
                let ident = &text[..byte_len];

                if ident.trim_start_matches(['_', '.']).len() == 0 {
                    Some(ReadTokenResult {
                        token: TokenKind::Error {
                            message: ErrorTokenMessage::new(
                                "identifier only contains underscores and dots",
                                0,
                                byte_len,
                            ),
                            hint_message: None,
                            dummy: Some(TokenKind::dummy_identifier()),
                        },
                        consumed_bytes: byte_len,
                    })
                } else {
                    for (kw_str, kw) in KEYWORD_MAP.iter().copied() {
                        if ident.eq_ignore_ascii_case(kw_str) {
                            return Some(ReadTokenResult {
                                token: TokenKind::Keyword(kw),
                                consumed_bytes: byte_len,
                            });
                        }
                    }

                    for (r_str, r) in REGISTER_MAP.iter().copied() {
                        if ident.eq_ignore_ascii_case(r_str) {
                            return Some(ReadTokenResult {
                                token: TokenKind::Register(r),
                                consumed_bytes: byte_len,
                            });
                        }
                    }

                    Some(ReadTokenResult {
                        token: TokenKind::Identifier(ident.into()),
                        consumed_bytes: byte_len,
                    })
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl langbox::TokenReader for TokenReader {
    type Token = TokenKind;

    fn read_token(text: &str) -> ReadTokenResult {
        let mut result: Option<ReadTokenResult> = None;

        macro_rules! try_read {
            ($f:ident) => {
                if result.is_none() {
                    result = Self::$f(text);
                }
            };
        }

        try_read!(read_new_line);
        try_read!(read_line_comment);
        try_read!(read_block_comment);
        try_read!(read_operator);
        try_read!(read_integer_literal);
        try_read!(read_char_literal);
        try_read!(read_string_literal);
        try_read!(read_directive);
        try_read!(read_identifier);

        if let Some(result) = result {
            result
        } else {
            let byte_len = text.chars().next().map(|c| c.len_utf8()).unwrap_or(0);

            ReadTokenResult {
                token: TokenKind::Error {
                    message: ErrorTokenMessage::new("unexpected character", 0, byte_len),
                    hint_message: None,
                    dummy: None,
                },
                consumed_bytes: byte_len,
            }
        }
    }
}

#[cfg(test)]
fn test_lexer(input: &'static str, expected: &[TokenKind]) {
    use langbox::*;
    use termcolor::*;

    let stdout = StandardStream::stdout(ColorChoice::Auto);
    let mut stdout = stdout.lock();

    let mut file_server = FileServer::new();
    let file = file_server.register_file_memory("<test>", input);

    let mut lexer =
        Lexer::<self::TokenReader, whitespace_mode::RemoveKeepNewLine>::new(file, &file_server);

    let mut tokens = Vec::new();
    let mut has_errors = false;
    loop {
        match lexer.next() {
            Some(Token { kind, .. }) => {
                if let TokenKind::Error { message, .. } = kind {
                    message
                        .pretty_print(&mut stdout, file, &file_server, crate::MessageKind::Error)
                        .unwrap();
                    has_errors = true;
                } else {
                    tokens.push(kind)
                }
            }
            None => break,
        }
    }

    if has_errors {
        panic!();
    }

    assert_eq!(tokens.len(), expected.len());
    for (t, e) in tokens.iter().zip(expected.iter()) {
        assert_eq!(t, e);
    }
}

#[test]
fn parses_empty_input() {
    test_lexer("", &[]);
}

#[test]
fn parses_new_line() {
    test_lexer("\n", &[TokenKind::NewLine]);
}

#[test]
fn parses_line_comment() {
    test_lexer(
        "// comment",
        &[TokenKind::Comment {
            has_new_line: false,
        }],
    );

    test_lexer(
        "// comment\n",
        &[
            TokenKind::Comment {
                has_new_line: false,
            },
            TokenKind::NewLine,
        ],
    );

    test_lexer(
        "label: // comment\n",
        &[
            TokenKind::Identifier("label".into()),
            TokenKind::Operator(Operator::Colon),
            TokenKind::Comment {
                has_new_line: false,
            },
            TokenKind::NewLine,
        ],
    );

    test_lexer(
        "nop // comment\n",
        &[
            TokenKind::Keyword(Keyword::Nop),
            TokenKind::Comment {
                has_new_line: false,
            },
            TokenKind::NewLine,
        ],
    );
}

#[test]
fn parses_block_comment() {
    test_lexer(
        "/*comment*//* comment\ncomment */\n/*comment*/",
        &[
            TokenKind::Comment {
                has_new_line: false,
            },
            TokenKind::Comment { has_new_line: true },
            TokenKind::NewLine,
            TokenKind::Comment {
                has_new_line: false,
            },
        ],
    );

    test_lexer(
        "label: /*comment*/",
        &[
            TokenKind::Identifier("label".into()),
            TokenKind::Operator(Operator::Colon),
            TokenKind::Comment {
                has_new_line: false,
            },
        ],
    );

    test_lexer(
        "nop /*comment*/",
        &[
            TokenKind::Keyword(Keyword::Nop),
            TokenKind::Comment {
                has_new_line: false,
            },
        ],
    );
}

#[test]
fn parses_operators() {
    test_lexer(
        ">>>+<<>,,",
        &[
            TokenKind::Operator(Operator::ShiftRightArithmetic),
            TokenKind::Operator(Operator::Plus),
            TokenKind::Operator(Operator::ShiftLeft),
            TokenKind::Operator(Operator::GreaterThan),
            TokenKind::Operator(Operator::Comma),
            TokenKind::Operator(Operator::Comma),
        ],
    );
}

#[test]
fn parses_integer_literals() {
    test_lexer(
        "0b101\n0o12\n0xF\n20",
        &[
            TokenKind::IntegerLiteral(5),
            TokenKind::NewLine,
            TokenKind::IntegerLiteral(10),
            TokenKind::NewLine,
            TokenKind::IntegerLiteral(15),
            TokenKind::NewLine,
            TokenKind::IntegerLiteral(20),
        ],
    );
}

#[test]
fn parses_char_literals() {
    test_lexer(
        "'c''_''0'",
        &[
            TokenKind::CharLiteral('c'),
            TokenKind::CharLiteral('_'),
            TokenKind::CharLiteral('0'),
        ],
    );
}

#[test]
fn parses_char_literal_escapes() {
    test_lexer(
        "'\\\\''\\n''\\0'",
        &[
            TokenKind::CharLiteral('\\'),
            TokenKind::CharLiteral('\n'),
            TokenKind::CharLiteral('\0'),
        ],
    );
}

#[test]
fn parses_char_literal_hex_escapes() {
    test_lexer(
        "'\\x20''\\u0020'",
        &[TokenKind::CharLiteral(' '), TokenKind::CharLiteral(' ')],
    );
}

#[test]
fn parses_string_literals() {
    test_lexer("\"string\"", &[TokenKind::StringLiteral("string".into())]);
}

#[test]
fn parses_string_literal_escapes() {
    test_lexer(
        "\"\\\\\\n\\0\"",
        &[TokenKind::StringLiteral("\\\n\0".into())],
    );
}

#[test]
fn parses_string_literal_hex_escapes() {
    test_lexer("\"\\x20\\u0020\"", &[TokenKind::StringLiteral("  ".into())]);
}

#[test]
fn parses_directives() {
    test_lexer("#include", &[TokenKind::Directive(Directive::Include)]);
}

#[test]
fn parses_keywords() {
    test_lexer(
        "add\naddc\nbr.c",
        &[
            TokenKind::Keyword(Keyword::Add),
            TokenKind::NewLine,
            TokenKind::Keyword(Keyword::AddC),
            TokenKind::NewLine,
            TokenKind::Keyword(Keyword::BrC),
        ],
    );
}

#[test]
fn parses_registers() {
    test_lexer(
        "r15\nzero\na0",
        &[
            TokenKind::Register(Register(u5!(15))),
            TokenKind::NewLine,
            TokenKind::Register(Register::ZERO),
            TokenKind::NewLine,
            TokenKind::Register(Register::A0),
        ],
    );
}

#[test]
fn parses_identifiers() {
    test_lexer(
        "foo\n_bar\nbaz_\n_0\na.0",
        &[
            TokenKind::Identifier("foo".into()),
            TokenKind::NewLine,
            TokenKind::Identifier("_bar".into()),
            TokenKind::NewLine,
            TokenKind::Identifier("baz_".into()),
            TokenKind::NewLine,
            TokenKind::Identifier("_0".into()),
            TokenKind::NewLine,
            TokenKind::Identifier("a.0".into()),
        ],
    );
}
