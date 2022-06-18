use crate::{Message, MessageKind, OptionalResult, Register};
use std::borrow::Cow;
use std::iter::Peekable;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::pattern::Pattern;
use std::str::CharIndices;

pub struct InputFile {
    path: PathBuf,
    text: Cow<'static, str>,
}
impl InputFile {
    pub fn new_from_memory<T: Into<Cow<'static, str>>>(name: &str, text: T) -> Rc<Self> {
        Rc::new(Self {
            path: name.into(),
            text: text.into(),
        })
    }
}

#[derive(Clone)]
pub struct TextSpan {
    file: Rc<InputFile>,
    line: usize,
    column: usize,
    byte_start: usize,
    byte_end: usize,
}
impl TextSpan {
    #[inline]
    pub fn file_path(&self) -> &Path {
        &self.file.path
    }

    #[inline]
    pub fn line(&self) -> usize {
        self.line
    }

    #[inline]
    pub fn column(&self) -> usize {
        self.column
    }

    #[inline]
    pub fn text(&self) -> &str {
        &self.file.text[self.byte_start..self.byte_end]
    }

    pub fn split_into_lines(&self) -> Vec<Self> {
        use line_span::*;
        let mut result = Vec::new();

        let mut byte_start = self.byte_start;
        let mut line_offset = 0;
        let mut column = self.column;
        loop {
            let byte_end = find_line_end(&self.file.text, byte_start);
            result.push(Self {
                file: Rc::clone(&self.file),
                line: self.line + line_offset,
                column,
                byte_start,
                byte_end: byte_end.min(self.byte_end),
            });

            if byte_end < self.byte_end {
                if let Some(next_start) = find_next_line_start(&self.file.text, byte_end) {
                    byte_start = next_start;
                    line_offset += 1;
                    column = 0;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        result
    }

    pub fn enclosing_lines(&self) -> Vec<Self> {
        use line_span::*;
        let mut result = Vec::new();

        let mut byte_start = find_line_start(&self.file.text, self.byte_start);
        let mut line_offset = 0;
        loop {
            let byte_end = find_line_end(&self.file.text, byte_start);
            result.push(Self {
                file: Rc::clone(&self.file),
                line: self.line + line_offset,
                column: 0,
                byte_start,
                byte_end,
            });

            if byte_end < self.byte_end {
                if let Some(next_start) = find_next_line_start(&self.file.text, byte_end) {
                    byte_start = next_start;
                    line_offset += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        result
    }

    pub fn subspan(&self, column_offset: usize, char_count: usize, byte_offset: usize) -> Self {
        let byte_start = self.byte_start + byte_offset;

        let mut iter = self.file.text[byte_start..self.byte_end].char_indices();
        let (byte_len, _) = iter.nth(char_count).expect("Out of parent range");

        Self {
            file: Rc::clone(&self.file),
            line: self.line,
            column: self.column + column_offset,
            byte_start,
            byte_end: byte_start + byte_len,
        }
    }

    pub fn combine(&self, other: &Self) -> Self {
        let start_span = if self.byte_start < other.byte_start {
            self
        } else {
            other
        };

        Self {
            file: Rc::clone(&self.file),
            line: start_span.line,
            column: start_span.column,
            byte_start: start_span.byte_start,
            byte_end: self.byte_end.max(other.byte_end),
        }
    }
}
impl std::fmt::Debug for TextSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{} \"{}\"",
            self.file_path().display(),
            self.line(),
            self.column(),
            self.text()
        )
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

    Mov,
    LdI,
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
    Bra,

    LdUi,
    AddPcUi,

    Sys,
    ClrK,
}

#[rustfmt::skip]
const KEYWORD_MAP: &[(&str, Keyword)] = &[
    ("nop"    , Keyword::Nop    ),
    ("brk"    , Keyword::Brk    ),
    ("hlt"    , Keyword::Hlt    ),
    ("err"    , Keyword::Err    ),

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

    ("mov"    , Keyword::Mov    ),
    ("ldi"    , Keyword::LdI    ),
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
    ("bra"    , Keyword::Bra    ),

    ("ldui"   , Keyword::LdUi   ),
    ("addpcui", Keyword::AddPcUi),

    ("sys"    , Keyword::Sys    ),
    ("clrk"   , Keyword::ClrK   ),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    NewLine,
    Whitespace,
    Comment { has_new_line: bool },
    Operator(Operator),
    IntegerLiteral(i64),
    CharLiteral(char),
    StringLiteral(String),
    Directive(Directive),
    Register(Register),
    Keyword(Keyword),
    Identifier(String),
}
impl TokenKind {
    #[inline]
    fn dummy_integer() -> Self {
        Self::IntegerLiteral(0)
    }

    #[inline]
    fn dummy_char() -> Self {
        Self::CharLiteral('\0')
    }

    #[inline]
    fn dummy_string() -> Self {
        Self::StringLiteral("".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    span: TextSpan,
    kind: TokenKind,
}
impl Token {
    #[inline]
    pub fn span(&self) -> &TextSpan {
        &self.span
    }

    #[inline]
    pub fn kind(&self) -> &TokenKind {
        &self.kind
    }
}

#[derive(Debug)]
pub struct LexerError {
    dummy_token: Token,
    error_message: Message,
    hint_message: Option<Message>,
}
impl LexerError {
    fn new<S>(dummy_token: Token, error_span: TextSpan, error_text: S) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        let error_message = Message {
            kind: MessageKind::Error,
            token_span: dummy_token.span().clone(),
            span: error_span,
            text: error_text.into(),
        };

        Self {
            dummy_token,
            error_message,
            hint_message: None,
        }
    }

    fn new_with_hint<S1, S2>(
        dummy_token: Token,
        error_span: TextSpan,
        error_text: S1,
        hint_span: TextSpan,
        hint_text: S2,
    ) -> Self
    where
        S1: Into<Cow<'static, str>>,
        S2: Into<Cow<'static, str>>,
    {
        let error_message = Message {
            kind: MessageKind::Error,
            token_span: dummy_token.span().clone(),
            span: error_span,
            text: error_text.into(),
        };

        let hint_message = Message {
            kind: MessageKind::Hint,
            token_span: dummy_token.span().clone(),
            span: hint_span,
            text: hint_text.into(),
        };

        Self {
            dummy_token,
            error_message,
            hint_message: Some(hint_message),
        }
    }

    #[inline]
    pub fn into_dummy_token(self) -> Token {
        self.dummy_token
    }

    pub fn pretty_print<W: termcolor::WriteColor + std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        self.error_message.pretty_print(writer)?;

        if let Some(hint_message) = &self.hint_message {
            hint_message.pretty_print(writer)?;
        }

        Ok(())
    }
}

pub type LexerResult = OptionalResult<Token, LexerError>;

fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r')
}

fn is_bin_digit(c: char) -> bool {
    matches!(c, '0' | '1')
}

fn is_oct_digit(c: char) -> bool {
    matches!(c, '0'..='7')
}

fn is_dec_digit(c: char) -> bool {
    matches!(c, '0'..='9')
}

fn is_hex_digit(c: char) -> bool {
    matches!(c, '0'..='9' | 'a'..='f' | 'A'..='F')
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() | (c == '_') | (c == '.')
}

struct TextInput {
    file: Rc<InputFile>,
    next_byte_pos: usize,
    next_line: usize,
    next_column: usize,
}
impl TextInput {
    fn new(file: &Rc<InputFile>) -> Self {
        Self {
            file: Rc::clone(file),
            next_byte_pos: 0,
            next_line: 0,
            next_column: 0,
        }
    }

    #[inline]
    fn file(&self) -> &Rc<InputFile> {
        &self.file
    }

    #[inline]
    fn next_byte_pos(&self) -> usize {
        self.next_byte_pos
    }

    #[inline]
    fn next_line(&self) -> usize {
        self.next_line
    }

    #[inline]
    fn next_column(&self) -> usize {
        self.next_column
    }

    #[inline]
    fn remaining(&self) -> &str {
        &self.file.text[self.next_byte_pos..]
    }

    fn peek_char(&mut self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        self.remaining().starts_with(pat)
    }

    fn next_char(&mut self) -> Option<char> {
        let mut iter = self.remaining().char_indices();

        if let Some((_, c)) = iter.next() {
            self.next_byte_pos = if let Some((i, _)) = iter.next() {
                self.next_byte_pos + i
            } else {
                self.file.text.len()
            };

            if c == '\n' {
                self.next_line += 1;
                self.next_column = 0;
            } else {
                self.next_column += 1;
            }

            Some(c)
        } else {
            None
        }
    }

    fn advance_by(&mut self, bytes: usize) {
        let mut iter = self.file.text[self.next_byte_pos..].char_indices();

        while let Some((p, c)) = iter.next() {
            assert!(p <= bytes);

            if p == bytes {
                break;
            }

            if c == '\n' {
                self.next_line += 1;
                self.next_column = 0;
            } else {
                self.next_column += 1;
            }
        }

        self.next_byte_pos += bytes;
    }

    fn advance_while(
        &mut self,
        mut predicate: impl FnMut(char) -> bool,
        mut output: Option<&mut String>,
    ) -> usize {
        let mut iter = self.file.text[self.next_byte_pos..]
            .char_indices()
            .peekable();
        let mut count = 0;

        while let Some((i, c)) = iter.peek().copied() {
            if predicate(c) {
                if let Some(output) = &mut output {
                    output.push(c);
                }

                if c == '\n' {
                    self.next_line += 1;
                    self.next_column = 0;
                } else {
                    self.next_column += 1;
                }

                iter.next();
                count += 1;
            } else {
                self.next_byte_pos += i;
                return count;
            }
        }

        self.next_byte_pos = self.file.text.len();
        count
    }
}

fn next_n<I: Iterator>(iter: &mut I, n: usize, map: impl Fn(I::Item) -> char) -> Option<String> {
    let mut result = String::with_capacity(n);

    for _ in 0..n {
        if let Some(c) = iter.next() {
            result.push(map(c));
        } else {
            return None;
        }
    }

    Some(result)
}

fn parse_hex_escape(
    text: &str,
    iter: &mut Peekable<CharIndices>,
    s: &mut String,
    line: usize,
    column: usize,
    p: usize,
    span: &TextSpan,
    n: usize,
    kind_string: &str,
    count_string: &str,
    dummy_token: Token,
) -> Result<(), LexerError> {
    if let Some(digits) = next_n(iter, n, |(_, c)| c) {
        let end_p = iter.peek().map(|(ep, _)| *ep).unwrap_or(text.len());

        if let Ok(code) = u32::from_str_radix(&digits, 16) {
            if let Some(c) = char::from_u32(code) {
                s.push(c);
                Ok(())
            } else {
                let token_span = TextSpan {
                    file: Rc::clone(&span.file),
                    line,
                    column,
                    byte_start: p,
                    byte_end: end_p,
                };

                let hint_span = TextSpan {
                    file: Rc::clone(&span.file),
                    line,
                    column: column + 2,
                    byte_start: p + 2,
                    byte_end: end_p,
                };

                let err = LexerError::new_with_hint(
                    dummy_token,
                    token_span,
                    "invalid escape sequence",
                    hint_span,
                    format!("`{:X}` is not a valid codepoint", code),
                );

                Err(err)
            }
        } else {
            let token_span = TextSpan {
                file: Rc::clone(&span.file),
                line,
                column,
                byte_start: p,
                byte_end: end_p,
            };

            let hint_span = TextSpan {
                file: Rc::clone(&span.file),
                line,
                column: column + 2,
                byte_start: p + 2,
                byte_end: end_p,
            };

            let err = LexerError::new_with_hint(
                dummy_token,
                token_span,
                "invalid escape sequence",
                hint_span,
                "some characters are not valid hexadecimal digits",
            );

            Err(err)
        }
    } else {
        let end_p = iter.peek().map(|(ep, _)| *ep).unwrap_or(text.len());

        let token_span = TextSpan {
            file: Rc::clone(&span.file),
            line,
            column,
            byte_start: p,
            byte_end: end_p,
        };

        let err = LexerError::new_with_hint(
            dummy_token,
            token_span.clone(),
            "incomplete escape sequence",
            token_span,
            format!(
                "{} escape sequence must contain exactly {} hex digits",
                kind_string, count_string
            ),
        );

        Err(err)
    }
}

pub struct Lexer {
    input: TextInput,
    current_byte_pos: usize,
    current_line: usize,
    current_column: usize,
}
impl Lexer {
    pub fn new(file: &Rc<InputFile>) -> Self {
        Self {
            input: TextInput::new(file),
            current_byte_pos: 0,
            current_line: 0,
            current_column: 0,
        }
    }

    fn span(&self) -> TextSpan {
        TextSpan {
            file: Rc::clone(self.input.file()),
            line: self.current_line,
            column: self.current_column,
            byte_start: self.current_byte_pos,
            byte_end: self.input.next_byte_pos(),
        }
    }

    #[inline]
    fn get_token(&self, kind: TokenKind) -> Token {
        Token {
            span: self.span(),
            kind,
        }
    }

    fn next_new_line(&mut self) -> LexerResult {
        if let Some(c) = self.input.peek_char() {
            if c == '\n' {
                self.input.next_char();
                LexerResult::Some(self.get_token(TokenKind::NewLine))
            } else {
                LexerResult::None
            }
        } else {
            LexerResult::None
        }
    }

    fn next_whitespace(&mut self) -> LexerResult {
        let count = self.input.advance_while(is_whitespace, None);
        if count > 0 {
            LexerResult::Some(self.get_token(TokenKind::Whitespace))
        } else {
            LexerResult::None
        }
    }

    fn next_line_comment(&mut self) -> LexerResult {
        if self.input.starts_with("//") {
            self.input.advance_while(|c| c != '\n', None);
            LexerResult::Some(self.get_token(TokenKind::Comment {
                has_new_line: false,
            }))
        } else {
            LexerResult::None
        }
    }

    fn next_block_comment(&mut self) -> LexerResult {
        if self.input.starts_with("/*") {
            loop {
                self.input.advance_while(|c| c != '*', None);
                self.input.advance_while(|c| c == '*', None);

                if let Some(c) = self.input.peek_char() {
                    if c == '/' {
                        self.input.next_char();
                        break;
                    }
                } else {
                    let hint_span = self.span();

                    let error_span = TextSpan {
                        file: Rc::clone(&hint_span.file),
                        line: self.input.next_line(),
                        column: self.input.next_column(),
                        byte_start: self.input.file.text.len(),
                        byte_end: self.input.file.text.len(),
                    };

                    let err = LexerError::new_with_hint(
                        self.get_token(TokenKind::Comment {
                            has_new_line: self.current_line != self.input.next_line(),
                        }),
                        error_span,
                        "open block comment",
                        hint_span,
                        "block comments need to be closed with `*/`",
                    );

                    return LexerResult::Err(err);
                }
            }

            LexerResult::Some(self.get_token(TokenKind::Comment {
                has_new_line: self.current_line != self.input.next_line(),
            }))
        } else {
            LexerResult::None
        }
    }

    fn next_operator(&mut self) -> LexerResult {
        for (ops, op) in OPERATOR_MAP.iter().copied() {
            if self.input.starts_with(&ops) {
                self.input.advance_by(ops.len());

                return LexerResult::Some(self.get_token(TokenKind::Operator(op)));
            }
        }

        LexerResult::None
    }

    fn parse_integer(&mut self, span: TextSpan) -> Result<i64, LexerError> {
        let mut s = String::with_capacity(span.text().len());
        let mut iter = span.text().char_indices().peekable();
        let (_, first) = iter.next().unwrap();

        let is_valid: fn(char) -> bool;

        let radix;
        let mut column_offset = 0;
        if first == '0' {
            if let Some((_, prefix)) = iter.peek() {
                match prefix {
                    'b' | 'B' => {
                        iter.next();
                        radix = 2;
                        column_offset = 2;
                        is_valid = is_bin_digit;
                    }
                    'o' | 'O' => {
                        iter.next();
                        radix = 8;
                        column_offset = 2;
                        is_valid = is_oct_digit;
                    }
                    'x' | 'X' => {
                        iter.next();
                        radix = 16;
                        column_offset = 2;
                        is_valid = is_hex_digit;
                    }
                    _ => {
                        s.push(first);
                        radix = 10;
                        is_valid = is_dec_digit;
                    }
                }
            } else {
                return Ok(0);
            }
        } else {
            s.push(first);
            radix = 10;
            is_valid = is_dec_digit;
        }

        for (p, c) in iter {
            if c != '_' {
                if is_valid(c) {
                    s.push(c);
                } else {
                    let hint_span = span.subspan(column_offset, 1, p);

                    let err = LexerError::new_with_hint(
                        self.get_token(TokenKind::dummy_integer()),
                        span,
                        "illegal character in literal",
                        hint_span,
                        "this character is not valid in the literal",
                    );

                    return Err(err);
                }
            }

            column_offset += 1;
        }

        Ok(i64::from_str_radix(&s, radix).unwrap())
    }

    fn next_integer_literal(&mut self) -> LexerResult {
        if let Some(first) = self.input.peek_char() {
            if first.is_ascii_digit() {
                self.input.next_char();
                self.input.advance_while(is_ident_char, None);

                let span = self.span();
                match self.parse_integer(span) {
                    Ok(value) => {
                        LexerResult::Some(self.get_token(TokenKind::IntegerLiteral(value)))
                    }
                    Err(err) => LexerResult::Err(err),
                }
            } else {
                LexerResult::None
            }
        } else {
            LexerResult::None
        }
    }

    fn advance_input_until_char_escaped(&mut self, end_char: char) {
        let mut prev = None;
        self.input.advance_while(
            |c| {
                if let Some('\\') = prev {
                    prev = None;
                    true
                } else {
                    prev = Some(c);
                    c != end_char
                }
            },
            None,
        );
    }

    fn unescape_string(
        &mut self,
        span: TextSpan,
        dummy_token: Token,
    ) -> Result<String, LexerError> {
        let mut s = String::new();

        let text = &span.text()[1..(span.text().len() - 1)];
        let mut iter = text.char_indices().peekable();

        let mut line = span.line();
        let mut column = span.column() + 1;

        while let Some((p, c)) = iter.next() {
            if c == '\\' {
                let (next_p, next) = iter.next().expect("escape sequence invariant broken");

                match next {
                    '0' => {
                        s.push('\0');
                        column += 2;
                    }
                    'n' => {
                        s.push('\n');
                        column += 2;
                    }
                    'r' => {
                        s.push('\r');
                        column += 2;
                    }
                    't' => {
                        s.push('\t');
                        column += 2;
                    }
                    '\\' => {
                        s.push('\\');
                        column += 2;
                    }
                    '\'' => {
                        s.push('\'');
                        column += 2;
                    }
                    '\"' => {
                        s.push('\"');
                        column += 2;
                    }
                    'x' => {
                        parse_hex_escape(
                            text,
                            &mut iter,
                            &mut s,
                            line,
                            column,
                            p,
                            &span,
                            2,
                            "an ascii",
                            "two",
                            dummy_token.clone(),
                        )?;
                        column += 4;
                    }
                    'u' => {
                        parse_hex_escape(
                            text,
                            &mut iter,
                            &mut s,
                            line,
                            column,
                            p,
                            &span,
                            4,
                            "a unicode",
                            "four",
                            dummy_token.clone(),
                        )?;
                        column += 6;
                    }
                    _ => {
                        let end_p = iter.peek().map(|(ep, _)| *ep).unwrap_or(text.len());

                        let token_span = TextSpan {
                            file: Rc::clone(&span.file),
                            line,
                            column,
                            byte_start: p,
                            byte_end: end_p,
                        };

                        let hint_span = TextSpan {
                            file: Rc::clone(&span.file),
                            line,
                            column: column + 1,
                            byte_start: next_p,
                            byte_end: end_p,
                        };

                        let err = LexerError::new_with_hint(
                            dummy_token.clone(),
                            token_span,
                            "invalid escape sequence",
                            hint_span,
                            "valid escape sequences are `\\0`, `\\n`, `\\r`, `\\t`, `\\\\`, `\\'`, `\\\"`, `\\xXX` and `\\uUUUU`",
                        );

                        return Err(err);
                    }
                }
            } else {
                s.push(c);

                if c == '\n' {
                    line += 1;
                    column = 0;
                } else {
                    column += 1;
                }
            }
        }

        Ok(s)
    }

    fn next_char_literal(&mut self) -> LexerResult {
        if let Some(first) = self.input.peek_char() {
            if first == '\'' {
                self.input.next_char();
                self.advance_input_until_char_escaped('\'');

                if let Some(_) = self.input.next_char() {
                    match self.unescape_string(self.span(), self.get_token(TokenKind::dummy_char()))
                    {
                        Ok(s) => {
                            let mut iter = s.chars();
                            if let Some(c) = iter.next() {
                                if iter.next().is_some() {
                                    let err = LexerError::new(
                                        self.get_token(TokenKind::dummy_char()),
                                        self.span(),
                                        "char literal contains more than one codepoint",
                                    );

                                    LexerResult::Err(err)
                                } else {
                                    LexerResult::Some(self.get_token(TokenKind::CharLiteral(c)))
                                }
                            } else {
                                let err = LexerError::new(
                                    self.get_token(TokenKind::dummy_char()),
                                    self.span(),
                                    "empty char literal",
                                );

                                LexerResult::Err(err)
                            }
                        }
                        Err(err) => LexerResult::Err(err),
                    }
                } else {
                    let hint_span = self.span();

                    let error_span = TextSpan {
                        file: Rc::clone(&hint_span.file),
                        line: self.input.next_line(),
                        column: self.input.next_column(),
                        byte_start: self.input.file.text.len(),
                        byte_end: self.input.file.text.len(),
                    };

                    let err = LexerError::new_with_hint(
                        self.get_token(TokenKind::dummy_char()),
                        error_span,
                        "open char literal",
                        hint_span,
                        "char literals need to be closed with `'`",
                    );

                    return LexerResult::Err(err);
                }
            } else {
                LexerResult::None
            }
        } else {
            LexerResult::None
        }
    }

    fn next_string_literal(&mut self) -> LexerResult {
        if let Some(first) = self.input.peek_char() {
            if first == '"' {
                self.input.next_char();
                self.advance_input_until_char_escaped('"');

                if let Some(_) = self.input.next_char() {
                    match self
                        .unescape_string(self.span(), self.get_token(TokenKind::dummy_string()))
                    {
                        Ok(s) => LexerResult::Some(self.get_token(TokenKind::StringLiteral(s))),
                        Err(err) => LexerResult::Err(err),
                    }
                } else {
                    let hint_span = self.span();

                    let error_span = TextSpan {
                        file: Rc::clone(&hint_span.file),
                        line: self.input.next_line(),
                        column: self.input.next_column(),
                        byte_start: self.input.file.text.len(),
                        byte_end: self.input.file.text.len(),
                    };

                    let err = LexerError::new_with_hint(
                        self.get_token(TokenKind::dummy_string()),
                        error_span,
                        "open string literal",
                        hint_span,
                        "string literals need to be closed with `\"`",
                    );

                    return LexerResult::Err(err);
                }
            } else {
                LexerResult::None
            }
        } else {
            LexerResult::None
        }
    }

    fn next_directive(&mut self) -> LexerResult {
        if let Some(first) = self.input.peek_char() {
            if first == '#' {
                self.input.next_char();

                let mut s = String::new();
                self.input.advance_while(is_ident_char, Some(&mut s));

                if s.len() == 0 {
                    let hint_span = self.span();

                    let byte_end =
                        if let Some((p, _)) = self.input.remaining().char_indices().nth(1) {
                            hint_span.byte_end + p
                        } else {
                            self.input.file.text.len()
                        };

                    let error_span = TextSpan {
                        file: Rc::clone(&hint_span.file),
                        line: hint_span.line,
                        column: hint_span.column + 1,
                        byte_start: hint_span.byte_end,
                        byte_end,
                    };

                    let err = LexerError::new_with_hint(
                        self.get_token(TokenKind::Whitespace),
                        error_span,
                        "expected directive",
                        hint_span,
                        "`#` indicates the start of a directive",
                    );

                    LexerResult::Err(err)
                } else {
                    for (ds, d) in DIRECTIVE_MAP.iter().copied() {
                        if s.eq_ignore_ascii_case(ds) {
                            return LexerResult::Some(self.get_token(TokenKind::Directive(d)));
                        }
                    }

                    let err = LexerError::new(
                        self.get_token(TokenKind::Whitespace),
                        self.span(),
                        "unknown directive",
                    );

                    LexerResult::Err(err)
                }
            } else {
                LexerResult::None
            }
        } else {
            LexerResult::None
        }
    }

    fn next_identifier(&mut self) -> LexerResult {
        if let Some(first) = self.input.peek_char() {
            if first.is_alphabetic() | (first == '_') | (first == '.') {
                self.input.next_char();

                let mut s = String::new();
                s.push(first);
                self.input.advance_while(is_ident_char, Some(&mut s));

                if s.trim_start_matches('_').len() == 0 {
                    let err = LexerError::new(
                        self.get_token(TokenKind::Identifier(s)),
                        self.span(),
                        "identifier only contains underscores",
                    );

                    LexerResult::Err(err)
                } else if s.trim_start_matches('.').len() == 0 {
                    let err = LexerError::new(
                        self.get_token(TokenKind::Identifier(s)),
                        self.span(),
                        "identifier only contains dots",
                    );

                    LexerResult::Err(err)
                } else {
                    for (kws, kw) in KEYWORD_MAP.iter().copied() {
                        if s.eq_ignore_ascii_case(kws) {
                            return LexerResult::Some(self.get_token(TokenKind::Keyword(kw)));
                        }
                    }

                    for (rs, r) in REGISTER_MAP.iter().copied() {
                        if s.eq_ignore_ascii_case(rs) {
                            return LexerResult::Some(self.get_token(TokenKind::Register(r)));
                        }
                    }

                    LexerResult::Some(self.get_token(TokenKind::Identifier(s)))
                }
            } else {
                LexerResult::None
            }
        } else {
            LexerResult::None
        }
    }

    pub fn next_token(&mut self) -> LexerResult {
        if let Some(_) = self.input.peek_char() {
            let mut result = LexerResult::None;

            if result.is_none() {
                result = self.next_new_line();
            }

            if result.is_none() {
                result = self.next_whitespace();
            }

            if result.is_none() {
                result = self.next_line_comment();
            }

            if result.is_none() {
                result = self.next_block_comment();
            }

            if result.is_none() {
                result = self.next_operator();
            }

            if result.is_none() {
                result = self.next_integer_literal();
            }

            if result.is_none() {
                result = self.next_char_literal();
            }

            if result.is_none() {
                result = self.next_string_literal();
            }

            if result.is_none() {
                result = self.next_directive();
            }

            if result.is_none() {
                result = self.next_identifier();
            }

            if result.is_none() {
                self.input.next_char();

                result = LexerResult::Err(LexerError::new(
                    self.get_token(TokenKind::Whitespace),
                    self.span(),
                    "unexpected character",
                ));
            }

            self.current_byte_pos = self.input.next_byte_pos();
            self.current_line = self.input.next_line();
            self.current_column = self.input.next_column();

            result
        } else {
            LexerResult::None
        }
    }
}
impl Iterator for Lexer {
    type Item = Result<Token, LexerError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token().into()
    }
}

#[cfg(test)]
fn test_lexer(input: &'static str, expected: &[TokenKind]) {
    use termcolor::*;

    let stdout = StandardStream::stdout(ColorChoice::Auto);
    let mut stdout = stdout.lock();

    let file = InputFile::new_from_memory("test", input);
    let mut lexer = Lexer::new(&file);

    let mut tokens = Vec::new();
    let mut has_errors = false;
    loop {
        match lexer.next_token() {
            OptionalResult::Some(token) => tokens.push(token),
            OptionalResult::None => break,
            OptionalResult::Err(err) => {
                err.pretty_print(&mut stdout).unwrap();
                has_errors = true;
            }
        }
    }

    if has_errors {
        panic!();
    }

    assert_eq!(tokens.len(), expected.len());
    for (t, e) in tokens.iter().zip(expected.iter()) {
        assert_eq!(&t.kind, e);
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
fn parses_whitespace() {
    test_lexer("  \t  \r  ", &[TokenKind::Whitespace]);
}

#[test]
fn parses_multiple() {
    test_lexer(
        " \n ",
        &[
            TokenKind::Whitespace,
            TokenKind::NewLine,
            TokenKind::Whitespace,
        ],
    );
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
        "    // comment",
        &[
            TokenKind::Whitespace,
            TokenKind::Comment {
                has_new_line: false,
            },
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
    test_lexer(
        "\"string\"",
        &[TokenKind::StringLiteral("string".to_string())],
    );
}

#[test]
fn parses_string_literal_escapes() {
    test_lexer(
        "\"\\\\\\n\\0\"",
        &[TokenKind::StringLiteral("\\\n\0".to_string())],
    );
}

#[test]
fn parses_string_literal_hex_escapes() {
    test_lexer(
        "\"\\x20\\u0020\"",
        &[TokenKind::StringLiteral("  ".to_string())],
    );
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
            TokenKind::Register(Register(u5!(0))),
            TokenKind::NewLine,
            TokenKind::Register(Register(u5!(4))),
        ],
    );
}

#[test]
fn parses_identifiers() {
    test_lexer(
        "foo\n_bar\nbaz_\n_0\na.0",
        &[
            TokenKind::Identifier("foo".to_string()),
            TokenKind::NewLine,
            TokenKind::Identifier("_bar".to_string()),
            TokenKind::NewLine,
            TokenKind::Identifier("baz_".to_string()),
            TokenKind::NewLine,
            TokenKind::Identifier("_0".to_string()),
            TokenKind::NewLine,
            TokenKind::Identifier("a.0".to_string()),
        ],
    );
}
