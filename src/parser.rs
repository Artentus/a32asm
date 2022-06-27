use crate::lexer::*;
use crate::{Message, MessageKind, Register, SharedString};
use std::borrow::Cow;
use std::fmt::Display;
use std::rc::Rc;

#[derive(Clone, Copy)]
struct TokenInput<'a> {
    tokens: &'a [Token],
    position: usize,
}
impl<'a> TokenInput<'a> {
    #[inline]
    fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    #[inline]
    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.position)
    }

    #[inline]
    fn advance(&self) -> Self {
        Self {
            tokens: self.tokens,
            position: (self.position + 1).min(self.tokens.len()),
        }
    }

    fn error_span(&self, all: bool) -> TextSpan {
        if let Some(next) = self.peek() {
            if all {
                let last = self.tokens.last().unwrap();
                next.span().combine(last.span())
            } else {
                next.span().clone()
            }
        } else {
            let last = self.tokens.last().unwrap();
            last.span().clone()
        }
    }

    fn hint_span(&self, all: bool) -> TextSpan {
        if let Some(prev) = self.tokens.get(self.position.saturating_sub(1)) {
            if all {
                let first = self.tokens.first().unwrap();
                prev.span().combine(first.span())
            } else {
                prev.span().clone()
            }
        } else {
            let first = self.tokens.first().unwrap();
            first.span().clone()
        }
    }
}

pub struct ParseError {
    error_message: Message,
    hint_message: Option<Message>,
}
impl ParseError {
    fn new<S>(error_span: TextSpan, error_text: S) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        let error_message = Message {
            kind: MessageKind::Error,
            token_span: error_span.clone(),
            span: error_span,
            text: error_text.into(),
        };

        Self {
            error_message,
            hint_message: None,
        }
    }

    fn new_with_hint<S1, S2>(
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
            token_span: error_span.clone(),
            span: error_span,
            text: error_text.into(),
        };

        let hint_message = Message {
            kind: MessageKind::Hint,
            token_span: hint_span.clone(),
            span: hint_span,
            text: hint_text.into(),
        };

        Self {
            error_message,
            hint_message: Some(hint_message),
        }
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

macro_rules! error {
    ($error_msg:expr) => {
        |input| ParseError::new(input.error_span(false), $error_msg)
    };
    (all $error_msg:expr) => {
        |input| ParseError::new(input.error_span(true), $error_msg)
    };
    ($error_msg:expr, $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                input.error_span(false),
                $error_msg,
                input.hint_span(false),
                $hint_msg,
            )
        }
    };
    (all $error_msg:expr, $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                input.error_span(true),
                $error_msg,
                input.hint_span(false),
                $hint_msg,
            )
        }
    };
    ($error_msg:expr, all $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                input.error_span(false),
                $error_msg,
                input.hint_span(true),
                $hint_msg,
            )
        }
    };
    (all $error_msg:expr, all $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                input.error_span(true),
                $error_msg,
                input.hint_span(true),
                $hint_msg,
            )
        }
    };
}

#[must_use]
enum ParseResult<'a, T> {
    /// The input matched the parser pattern.
    Match {
        remaining: TokenInput<'a>,
        span: Option<TextSpan>,
        value: T,
    },
    /// The input did not match the parser pattern.
    NoMatch,
    /// The input matched the parser pattern but was malformed or invalid.
    Err(ParseError),
}
impl<'a, T> ParseResult<'a, T> {
    pub fn map<R>(self, f: impl FnOnce(T) -> R) -> ParseResult<'a, R> {
        match self {
            Self::Match {
                remaining,
                span,
                value,
            } => ParseResult::Match {
                remaining,
                span,
                value: f(value),
            },
            Self::NoMatch => ParseResult::NoMatch,
            Self::Err(err) => ParseResult::Err(err),
        }
    }
}

trait Parser<'a, T>: Fn(TokenInput<'a>) -> ParseResult<'a, T> + Clone {}
impl<'a, T, F> Parser<'a, T> for F where F: Fn(TokenInput<'a>) -> ParseResult<'a, T> + Clone {}

fn combine_spans(first: Option<TextSpan>, second: Option<TextSpan>) -> Option<TextSpan> {
    match (first, second) {
        (None, None) => None,
        (None, Some(second)) => Some(second),
        (Some(first), None) => Some(first),
        (Some(first), Some(second)) => Some(first.combine(&second)),
    }
}

fn and_then<'a, T1, T2>(
    first: impl Parser<'a, T1>,
    second: impl Parser<'a, T2>,
) -> impl Parser<'a, (T1, T2)> {
    move |input: TokenInput<'a>| match first(input) {
        ParseResult::Match {
            remaining,
            span: span1,
            value: v1,
        } => match second(remaining) {
            ParseResult::Match {
                remaining,
                span: span2,
                value: v2,
            } => {
                let span = combine_spans(span1, span2);
                ParseResult::Match {
                    remaining,
                    span,
                    value: (v1, v2),
                }
            }
            ParseResult::NoMatch => ParseResult::NoMatch,
            ParseResult::Err(err) => ParseResult::Err(err),
        },
        ParseResult::NoMatch => ParseResult::NoMatch,
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn or_else<'a, T>(first: impl Parser<'a, T>, second: impl Parser<'a, T>) -> impl Parser<'a, T> {
    move |input: TokenInput<'a>| match first(input) {
        ParseResult::Match {
            remaining,
            span,
            value,
        } => ParseResult::Match {
            remaining,
            span,
            value,
        },
        ParseResult::NoMatch => match second(input) {
            ParseResult::Match {
                remaining,
                span,
                value,
            } => ParseResult::Match {
                remaining,
                span,
                value,
            },
            ParseResult::NoMatch => ParseResult::NoMatch,
            ParseResult::Err(err) => ParseResult::Err(err),
        },
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn prefixed<'a, T1, T2>(
    first: impl Parser<'a, T1>,
    second: impl Parser<'a, T2>,
) -> impl Parser<'a, T2> {
    move |input: TokenInput<'a>| match first(input) {
        ParseResult::Match {
            remaining,
            span: span1,
            ..
        } => match second(remaining) {
            ParseResult::Match {
                remaining,
                span: span2,
                value: v2,
            } => {
                let span = combine_spans(span1, span2);
                ParseResult::Match {
                    remaining,
                    span,
                    value: v2,
                }
            }
            ParseResult::NoMatch => ParseResult::NoMatch,
            ParseResult::Err(err) => ParseResult::Err(err),
        },
        ParseResult::NoMatch => ParseResult::NoMatch,
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn suffixed<'a, T1, T2>(
    first: impl Parser<'a, T1>,
    second: impl Parser<'a, T2>,
) -> impl Parser<'a, T1> {
    move |input: TokenInput<'a>| match first(input) {
        ParseResult::Match {
            remaining,
            span: span1,
            value: v1,
        } => match second(remaining) {
            ParseResult::Match {
                remaining,
                span: span2,
                ..
            } => {
                let span = combine_spans(span1, span2);
                ParseResult::Match {
                    remaining,
                    span,
                    value: v1,
                }
            }
            ParseResult::NoMatch => ParseResult::NoMatch,
            ParseResult::Err(err) => ParseResult::Err(err),
        },
        ParseResult::NoMatch => ParseResult::NoMatch,
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn require<'a, T>(
    p: impl Parser<'a, T>,
    gen_err: impl Fn(TokenInput<'a>) -> ParseError + Clone,
) -> impl Parser<'a, T> {
    move |input: TokenInput<'a>| match p(input.clone()) {
        ParseResult::Match {
            remaining,
            span,
            value,
        } => ParseResult::Match {
            remaining,
            span,
            value,
        },
        ParseResult::NoMatch => ParseResult::Err(gen_err(input)),
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn verify<'a, T>(p: impl Parser<'a, T>, verify: impl Fn(&T) -> bool + Clone) -> impl Parser<'a, T> {
    move |input: TokenInput<'a>| match p(input.clone()) {
        ParseResult::Match {
            remaining,
            span,
            value,
        } => {
            if verify(&value) {
                ParseResult::Match {
                    remaining,
                    span,
                    value,
                }
            } else {
                ParseResult::NoMatch
            }
        }
        ParseResult::NoMatch => ParseResult::NoMatch,
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn or_return<'a, T: Clone>(p: impl Parser<'a, T>, default: T) -> impl Parser<'a, T> {
    move |input: TokenInput<'a>| match p(input.clone()) {
        ParseResult::Match {
            remaining,
            span,
            value,
        } => ParseResult::Match {
            remaining,
            span,
            value,
        },
        ParseResult::NoMatch => ParseResult::Match {
            remaining: input,
            span: None,
            value: default.clone(),
        },
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn map<'a, T, M>(p: impl Parser<'a, T>, f: impl Fn(T) -> M + Clone) -> impl Parser<'a, M> {
    move |input: TokenInput<'a>| p(input).map(&f)
}

fn map_to<'a, T, M: Clone>(p: impl Parser<'a, T>, v: M) -> impl Parser<'a, M> {
    move |input: TokenInput<'a>| p(input).map(|_| v.clone())
}

fn map_span<'a, T, M>(
    p: impl Parser<'a, T>,
    f: impl Fn(T, Option<TextSpan>) -> M + Clone,
) -> impl Parser<'a, M> {
    move |input: TokenInput<'a>| match p(input) {
        ParseResult::Match {
            remaining,
            span,
            value,
        } => ParseResult::Match {
            remaining,
            span: span.clone(),
            value: f(value, span),
        },
        ParseResult::NoMatch => ParseResult::NoMatch,
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

fn many<'a, T>(p: impl Parser<'a, T>, allow_empty: bool) -> impl Parser<'a, Vec<T>> {
    move |mut input: TokenInput<'a>| {
        let mut result = Vec::new();
        let mut full_span: Option<TextSpan> = None;

        loop {
            match p(input) {
                ParseResult::Match {
                    remaining,
                    span,
                    value,
                } => {
                    input = remaining;
                    result.push(value);
                    full_span = combine_spans(full_span, span);
                }
                ParseResult::NoMatch => break,
                ParseResult::Err(err) => return ParseResult::Err(err),
            }
        }

        if allow_empty || (result.len() > 0) {
            ParseResult::Match {
                remaining: input,
                span: full_span,
                value: result,
            }
        } else {
            ParseResult::NoMatch
        }
    }
}

fn sep_by<'a, T, S>(
    p: impl Parser<'a, T>,
    s: impl Parser<'a, S>,
    allow_empty: bool,
    allow_trailing: bool,
) -> impl Parser<'a, Vec<T>> {
    move |input: TokenInput<'a>| match p(input.clone()) {
        ParseResult::Match {
            remaining: mut input,
            span,
            value,
        } => {
            let mut result = Vec::new();
            result.push(value);

            let mut full_span = span;

            loop {
                match s(input) {
                    ParseResult::Match { remaining, .. } => match p(remaining) {
                        ParseResult::Match {
                            remaining,
                            span,
                            value,
                        } => {
                            input = remaining;
                            result.push(value);
                            full_span = combine_spans(full_span, span);
                        }
                        ParseResult::NoMatch => {
                            if allow_trailing {
                                break;
                            } else {
                                return ParseResult::NoMatch;
                            }
                        }
                        ParseResult::Err(err) => return ParseResult::Err(err),
                    },
                    ParseResult::NoMatch => break,
                    ParseResult::Err(err) => return ParseResult::Err(err),
                }
            }

            ParseResult::Match {
                remaining: input,
                span: full_span,
                value: result,
            }
        }
        ParseResult::NoMatch => {
            if allow_empty {
                ParseResult::Match {
                    remaining: input,
                    span: None,
                    value: Vec::new(),
                }
            } else {
                ParseResult::NoMatch
            }
        }
        ParseResult::Err(err) => ParseResult::Err(err),
    }
}

macro_rules! parser {
    ($p:expr) => {
        $p
    };
    ($lhs:expr, &, $rhs:expr, $($rest:tt)*) => {
        parser!(and_then($lhs, $rhs), $($rest)*)
    };
    ($lhs:expr, &, $rhs:expr) => {
        and_then($lhs, $rhs)
    };
    ($lhs:expr, <<, $rhs:expr, $($rest:tt)*) => {
        parser!(suffixed($lhs, $rhs), $($rest)*)
    };
    ($lhs:expr, <<, $rhs:expr) => {
        suffixed($lhs, $rhs)
    };
    ($lhs:expr, >>, $rhs:expr, $($rest:tt)*) => {
        parser!(prefixed($lhs, $rhs), $($rest)*)
    };
    ($lhs:expr, >>, $rhs:expr) => {
        prefixed($lhs, $rhs)
    };
    ($lhs:expr, |, $($rest:tt)*) => {
        or_else($lhs, parser!($($rest)*))
    };
    ($lhs:expr, |, $rhs:expr) => {
        or_else($lhs, $rhs)
    };
    ($lhs:expr, ->, $rhs:expr, $($rest:tt)*) => {
        parser!(map($lhs, $rhs), $($rest)*)
    };
    ($lhs:expr, ->, $rhs:expr) => {
        map($lhs, $rhs)
    };
    ($lhs:expr, =>, $rhs:expr, $($rest:tt)*) => {
        parser!(map_to($lhs, $rhs), $($rest)*)
    };
    ($lhs:expr, =>, $rhs:expr) => {
        map_to($lhs, $rhs)
    };
    ($lhs:expr, |->, $rhs:expr, $($rest:tt)*) => {
        parser!(map_span($lhs, $rhs), $($rest)*)
    };
    ($lhs:expr, |->, $rhs:expr) => {
        map_span($lhs, $rhs)
    };
}

macro_rules! choice {
    ($p:expr) => {
       $p
    };
    ($head:expr, $($tail:expr),+) => {
        or_else($head, choice!($($tail),+))
    };
}

fn whitespace(input: TokenInput) -> ParseResult<()> {
    if let Some(token) = input.peek() {
        if token.kind() == &TokenKind::Whitespace {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: (),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn comment(input: TokenInput) -> ParseResult<()> {
    if let Some(token) = input.peek() {
        if let TokenKind::Comment { .. } = token.kind() {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: (),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn operator<'a>(op: Operator) -> impl Parser<'a, ()> {
    move |input: TokenInput<'a>| {
        if let Some(token) = input.peek() {
            if token.kind() == &TokenKind::Operator(op) {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: Some(token.span().clone()),
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    }
}

fn integer_literal(input: TokenInput) -> ParseResult<i64> {
    if let Some(token) = input.peek() {
        if let TokenKind::IntegerLiteral(val) = token.kind() {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: *val,
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn char_literal(input: TokenInput) -> ParseResult<char> {
    if let Some(token) = input.peek() {
        if let TokenKind::CharLiteral(c) = token.kind() {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: *c,
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn string_literal(input: TokenInput) -> ParseResult<SharedString> {
    if let Some(token) = input.peek() {
        if let TokenKind::StringLiteral(s) = token.kind() {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: Rc::clone(s),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn directive<'a>(dir: Directive) -> impl Parser<'a, ()> {
    move |input: TokenInput<'a>| {
        if let Some(token) = input.peek() {
            if token.kind() == &TokenKind::Directive(dir) {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: Some(token.span().clone()),
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    }
}

fn register(input: TokenInput) -> ParseResult<Register> {
    if let Some(token) = input.peek() {
        if let TokenKind::Register(reg) = token.kind() {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: *reg,
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn keyword<'a>(kw: Keyword) -> impl Parser<'a, ()> {
    move |input: TokenInput<'a>| {
        if let Some(token) = input.peek() {
            if token.kind() == &TokenKind::Keyword(kw) {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: Some(token.span().clone()),
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    }
}

fn identifier(input: TokenInput) -> ParseResult<SharedString> {
    if let Some(token) = input.peek() {
        if let TokenKind::Identifier(ident) = token.kind() {
            ParseResult::Match {
                remaining: input.advance(),
                span: Some(token.span().clone()),
                value: Rc::clone(ident),
            }
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn eof(input: TokenInput) -> ParseResult<()> {
    if let Some(_) = input.peek() {
        ParseResult::NoMatch
    } else {
        ParseResult::Match {
            remaining: input,
            span: None,
            value: (),
        }
    }
}

fn whitespace0(input: TokenInput) -> ParseResult<()> {
    let inner = parser!(whitespace, |, comment);
    parser!(many(inner, true), =>, ())(input)
}

fn whitespace1(input: TokenInput) -> ParseResult<()> {
    let inner = parser!(whitespace, |, comment);
    parser!(many(inner, false), =>, ())(input)
}

fn in_brackets<'a, T>(p: impl Parser<'a, T>) -> impl Parser<'a, T> {
    move |input: TokenInput<'a>| {
        if let Some(token) = input.peek() {
            if token.kind() == &TokenKind::Operator(Operator::OpenBracket) {
                let hint_span = input.error_span(false);
                let remaining = input.advance();

                let closing = require(operator(Operator::CloseBracket), |input| {
                    ParseError::new_with_hint(
                        input.error_span(false),
                        "missing closing bracket",
                        hint_span.clone(),
                        "matching open bracket here",
                    )
                });

                let full_parser = parser!(whitespace0, >>, p.clone(), <<, whitespace0, <<, closing);
                full_parser(remaining)
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    }
}

fn label_ident(input: TokenInput) -> ParseResult<SharedString> {
    identifier(input)
}

fn const_ident(input: TokenInput) -> ParseResult<SharedString> {
    parser!(operator(Operator::Define), >>, require(identifier, error!(
        "expected identifier",
        "`$` indicates a constant identifier"
    )))(input)
}

fn comma_sep(input: TokenInput) -> ParseResult<()> {
    parser!(whitespace0, >>, operator(Operator::Comma), <<, whitespace0)(input)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Positive,
    Negative,
    Not,
}
impl Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOperator::Positive => Ok(()),
            UnaryOperator::Negative => write!(f, "-"),
            UnaryOperator::Not => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    ShiftLeft,
    ShiftRight,
    ShiftRightArithmetic,
    And,
    Or,
    Xor,
    Equals,
    NotEquals,
    LessEqual,
    Less,
    GreaterEqual,
    Greater,
}
impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Remainder => write!(f, "%"),
            BinaryOperator::ShiftLeft => write!(f, "<<"),
            BinaryOperator::ShiftRight => write!(f, ">>"),
            BinaryOperator::ShiftRightArithmetic => write!(f, ">>>"),
            BinaryOperator::And => write!(f, "&"),
            BinaryOperator::Or => write!(f, "|"),
            BinaryOperator::Xor => write!(f, "^"),
            BinaryOperator::Equals => write!(f, "=="),
            BinaryOperator::NotEquals => write!(f, "!="),
            BinaryOperator::LessEqual => write!(f, "<="),
            BinaryOperator::Less => write!(f, "<"),
            BinaryOperator::GreaterEqual => write!(f, ">="),
            BinaryOperator::Greater => write!(f, ">"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExpressionKind {
    IntegerConstant(i64),
    CharConstant(char),
    Label(SharedString),
    Define(SharedString),
    UnaryOperator(UnaryOperator, Box<Expression>),
    BinaryOperator(BinaryOperator, Box<Expression>, Box<Expression>),
    Parenthesized(Box<Expression>),
}
impl Display for ExpressionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IntegerConstant(val) => write!(f, "{}", val),
            Self::CharConstant(c) => write!(f, "{}", c),
            Self::Label(name) => write!(f, "{}", name),
            Self::Define(name) => write!(f, "${}", name),
            Self::UnaryOperator(op, sub_expr) => write!(f, "{}{}", op, sub_expr),
            Self::BinaryOperator(op, lhs_expr, rhs_expr) => {
                write!(f, "{} {} {}", lhs_expr, op, rhs_expr)
            }
            Self::Parenthesized(expr) => write!(f, "({})", expr),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expression {
    kind: ExpressionKind,
    span: Option<TextSpan>,
}
impl Expression {
    #[inline]
    const fn new_dummy_constant(val: i64) -> Self {
        Self {
            kind: ExpressionKind::IntegerConstant(val),
            span: None,
        }
    }

    fn new_integer(val: i64, span: Option<TextSpan>) -> Self {
        Self {
            kind: ExpressionKind::IntegerConstant(val),
            span,
        }
    }

    fn new_char(c: char, span: Option<TextSpan>) -> Self {
        Self {
            kind: ExpressionKind::CharConstant(c),
            span,
        }
    }

    fn new_label(name: SharedString, span: Option<TextSpan>) -> Self {
        Self {
            kind: ExpressionKind::Label(name),
            span,
        }
    }

    fn new_define(name: SharedString, span: Option<TextSpan>) -> Self {
        Self {
            kind: ExpressionKind::Define(name),
            span,
        }
    }

    fn new_unary(output: (UnaryOperator, Expression), span: Option<TextSpan>) -> Self {
        Self {
            kind: ExpressionKind::UnaryOperator(output.0, Box::new(output.1)),
            span,
        }
    }

    fn new_parenthesized(expr: Expression) -> Self {
        let span = expr.span().unwrap().clone();
        Self {
            kind: ExpressionKind::Parenthesized(Box::new(expr)),
            span: Some(span),
        }
    }

    #[inline]
    pub const fn kind(&self) -> &ExpressionKind {
        &self.kind
    }

    #[inline]
    pub const fn span(&self) -> Option<&TextSpan> {
        self.span.as_ref()
    }
}
impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.kind, f)
    }
}

fn leaf_expr(input: TokenInput) -> ParseResult<Expression> {
    choice!(
        paren_expr,
        parser!(integer_literal, |->, Expression::new_integer),
        parser!(char_literal, |->, Expression::new_char),
        parser!(label_ident, |->, Expression::new_label),
        parser!(const_ident, |->, Expression::new_define)
    )(input)
}

fn unary_op(input: TokenInput) -> ParseResult<UnaryOperator> {
    choice!(
        parser!(operator(Operator::Plus), =>, UnaryOperator::Positive),
        parser!(operator(Operator::Minus), =>, UnaryOperator::Negative),
        parser!(operator(Operator::Not), =>, UnaryOperator::Not)
    )(input)
}

fn unary_expr(input: TokenInput) -> ParseResult<Expression> {
    let unary = parser!(parser!(unary_op, <<, whitespace0, &, require(leaf_expr, error!("expected expression"))), |->, Expression::new_unary);
    choice!(leaf_expr, unary)(input)
}

fn aggregate_exprs<'a>(output: (Expression, Vec<(BinaryOperator, Expression)>)) -> Expression {
    let mut expr = output.0;
    for (op, rhs) in output.1 {
        let span = expr.span().unwrap().combine(rhs.span().unwrap());

        expr = Expression {
            kind: ExpressionKind::BinaryOperator(op, Box::new(expr), Box::new(rhs)),
            span: Some(span),
        };
    }
    expr
}

fn mul_div_op(input: TokenInput) -> ParseResult<BinaryOperator> {
    choice!(
        parser!(operator(Operator::Times), =>, BinaryOperator::Multiply),
        parser!(operator(Operator::Divide), =>, BinaryOperator::Divide),
        parser!(operator(Operator::Remainder), =>, BinaryOperator::Remainder)
    )(input)
}

fn mul_div_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, mul_div_op, <<, whitespace0, &, require(unary_expr, error!("expected expression")));
    map(parser!(unary_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn add_sub_op(input: TokenInput) -> ParseResult<BinaryOperator> {
    choice!(
        parser!(operator(Operator::Plus), =>, BinaryOperator::Add),
        parser!(operator(Operator::Minus), =>, BinaryOperator::Subtract)
    )(input)
}

fn add_sub_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, add_sub_op, <<, whitespace0, &, require(mul_div_expr, error!("expected expression")));
    map(parser!(mul_div_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn shift_op(input: TokenInput) -> ParseResult<BinaryOperator> {
    choice!(
        parser!(operator(Operator::ShiftLeft), =>, BinaryOperator::ShiftLeft),
        parser!(operator(Operator::ShiftRight), =>, BinaryOperator::ShiftRight),
        parser!(operator(Operator::ShiftRightArithmetic), =>, BinaryOperator::ShiftRightArithmetic)
    )(input)
}

fn shift_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, shift_op, <<, whitespace0, &, require(add_sub_expr, error!("expected expression")));
    map(parser!(add_sub_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn comp_op(input: TokenInput) -> ParseResult<BinaryOperator> {
    choice!(
        parser!(operator(Operator::LessThan), =>, BinaryOperator::Less),
        parser!(operator(Operator::LessThanEquals), =>, BinaryOperator::LessEqual),
        parser!(operator(Operator::GreaterThan), =>, BinaryOperator::Greater),
        parser!(operator(Operator::GreaterThanEquals), =>, BinaryOperator::GreaterEqual)
    )(input)
}

fn comp_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, comp_op, <<, whitespace0, &, require(shift_expr, error!("expected expression")));
    map(parser!(shift_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn eq_op(input: TokenInput) -> ParseResult<BinaryOperator> {
    choice!(
        parser!(operator(Operator::Equals), =>, BinaryOperator::Equals),
        parser!(operator(Operator::NotEquals), =>, BinaryOperator::NotEquals)
    )(input)
}

fn eq_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, eq_op, <<, whitespace0, &, require(comp_expr, error!("expected expression")));
    map(parser!(comp_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn and_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, map_to(operator(Operator::And), BinaryOperator::And), <<, whitespace0, &, require(eq_expr, error!("expected expression")));
    map(parser!(eq_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn xor_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, map_to(operator(Operator::Xor), BinaryOperator::Xor), <<, whitespace0, &, require(and_expr, error!("expected expression")));
    map(parser!(and_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn or_expr(input: TokenInput) -> ParseResult<Expression> {
    let tail = parser!(whitespace0, >>, map_to(operator(Operator::Or), BinaryOperator::Or), <<, whitespace0, &, require(xor_expr, error!("expected expression")));
    map(parser!(xor_expr, &, many(tail, true)), aggregate_exprs)(input)
}

fn paren_expr(input: TokenInput) -> ParseResult<Expression> {
    if let Some(token) = input.peek() {
        if token.kind() == &TokenKind::Operator(Operator::OpenParen) {
            let hint_span = input.error_span(false);
            let remaining = input.advance();

            let closing = require(operator(Operator::CloseParen), move |input| {
                ParseError::new_with_hint(
                    input.error_span(false),
                    "missing closing parenthesis",
                    hint_span.clone(),
                    "matching open paranthesis here",
                )
            });

            let full_parser = parser!(whitespace0, >>, require(or_expr, error!("expected expression")), <<, whitespace0, <<, closing);
            parser!(full_parser, ->, Expression::new_parenthesized)(remaining)
        } else {
            ParseResult::NoMatch
        }
    } else {
        ParseResult::NoMatch
    }
}

fn expr(input: TokenInput) -> ParseResult<Expression> {
    or_expr(input)
}

fn label_def(input: TokenInput) -> ParseResult<SharedString> {
    let colon = require(
        parser!(whitespace0, &, operator(Operator::Colon)),
        error!("expected `:`", "label declarations require a colon"),
    );
    parser!(identifier ,<<, colon)(input)
}

fn const_def(input: TokenInput) -> ParseResult<(SharedString, Expression)> {
    let assign = parser!(whitespace0, >>, require(operator(Operator::Assign), error!("expected assignment")));
    parser!(const_ident, <<, assign, <<, whitespace0, &, require(expr, error!("expected expression")))(
        input,
    )
}

fn display_expr_list(list: &[Expression]) -> std::result::Result<SharedString, std::fmt::Error> {
    use std::fmt::Write;

    let mut s = String::new();
    let mut iter = list.iter();

    write!(s, "{}", iter.next().unwrap())?;
    for expr in iter {
        write!(s, ", {}", expr)?;
    }

    Ok(s.into())
}

#[derive(Debug, Clone)]
pub enum AssemblerDirective {
    Include(SharedString),
    Address(u32),
    Align(u32),
    Int8(Vec<Expression>),
    Int16(Vec<Expression>),
    Int32(Vec<Expression>),
    Int64(Vec<Expression>),
    Ascii(SharedString),
    AsciiZ(SharedString),
    Utf8(SharedString),
    Utf16(SharedString),
    Unicode(SharedString),
}
impl Display for AssemblerDirective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Include(path) => write!(f, "#include \"{}\"", path),
            Self::Address(addr) => write!(f, "#address 0x{:0>8X}", addr),
            Self::Align(align) => write!(f, "#align {}", align),
            Self::Int8(vals) => write!(f, "#d8 {}", display_expr_list(vals)?),
            Self::Int16(vals) => write!(f, "#d16 {}", display_expr_list(vals)?),
            Self::Int32(vals) => write!(f, "#d32 {}", display_expr_list(vals)?),
            Self::Int64(vals) => write!(f, "#d64 {}", display_expr_list(vals)?),
            Self::Ascii(s) => write!(f, "#ascii \"{}\"", s),
            Self::AsciiZ(s) => write!(f, "#asciiz \"{}\"", s),
            Self::Utf8(s) => write!(f, "#utf8 \"{}\"", s),
            Self::Utf16(s) => write!(f, "#utf16 \"{}\"", s),
            Self::Unicode(s) => write!(f, "#unicode \"{}\"", s),
        }
    }
}

fn inc_dir(input: TokenInput) -> ParseResult<AssemblerDirective> {
    map(
        parser!(directive(Directive::Include), >>, require(whitespace1, error!(
            "expected whitespace",
            "whitespace is required between the directive and the file path"
        )), >>, require(string_literal, error!("expected file path"))),
        |s| AssemblerDirective::Include(s),
    )(input)
}

fn addr_dir(input: TokenInput) -> ParseResult<AssemblerDirective> {
    let addr_literal = verify(integer_literal, |addr| {
        (*addr <= (u32::MAX as i64)) && (*addr >= 0)
    });

    map(
        parser!(directive(Directive::Address), >>, require(whitespace1, error!(
            "expected whitespace",
            "whitespace is required between the directive and the address"
        )), >>, require(addr_literal, error!("expected address"))),
        |addr| AssemblerDirective::Address(addr as u32),
    )(input)
}

fn align_dir(input: TokenInput) -> ParseResult<AssemblerDirective> {
    let align_literal = verify(integer_literal, |align| {
        (*align <= (u32::MAX as i64)) && (*align >= 1)
    });

    map(
        parser!(directive(Directive::Align), >>, require(whitespace1, error!(
            "expected whitespace",
            "whitespace is required between the directive and the alignment"
        )), >>, require(align_literal, error!("expected alignment"))),
        |align| AssemblerDirective::Align(align as u32),
    )(input)
}

macro_rules! int_dir {
    ($name:ident, $dir:ident) => {
        fn $name(input: TokenInput) -> ParseResult<AssemblerDirective> {
            map(parser!(directive(Directive::$dir), >>, require(whitespace1, error!(
                    "expected whitespace",
                    "whitespace is required between the directive and the data"
                )), >>, require(sep_by(expr, comma_sep, false, true), error!("expected data")))
            , AssemblerDirective::$dir)(input)
        }
    };
}

int_dir!(int8_dir, Int8);
int_dir!(int16_dir, Int16);
int_dir!(int32_dir, Int32);
int_dir!(int64_dir, Int64);

macro_rules! string_dir {
    ($name:ident, $dir:ident) => {
        fn $name(input: TokenInput) -> ParseResult<AssemblerDirective> {
            map(parser!(directive(Directive::$dir), >>, require(whitespace1, error!(
                    "expected whitespace",
                    "whitespace is required between the directive and the string"
                )), >>, require(string_literal, error!("expected string")))
            , AssemblerDirective::$dir)(input)
        }
    };
}

string_dir!(ascii_dir, Ascii);
string_dir!(asciiz_dir, AsciiZ);
string_dir!(utf8_dir, Utf8);
string_dir!(utf16_dir, Utf16);
string_dir!(unicode_dir, Unicode);

fn dir<'a>(input: TokenInput) -> ParseResult<AssemblerDirective> {
    choice!(
        inc_dir,
        addr_dir,
        align_dir,
        int8_dir,
        int16_dir,
        int32_dir,
        int64_dir,
        ascii_dir,
        asciiz_dir,
        utf8_dir,
        utf16_dir,
        unicode_dir
    )(input)
}

#[derive(Debug, Clone)]
pub enum AluRhs {
    Register(Register),
    Immediate(Expression),
}
impl Display for AluRhs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AluRhs::Register(reg) => write!(f, "{}", reg),
            AluRhs::Immediate(imm) => write!(f, "{}", imm),
        }
    }
}

fn alu_rhs(input: TokenInput) -> ParseResult<AluRhs> {
    parser!(map(register, AluRhs::Register), |, map(expr, AluRhs::Immediate))(input)
}

#[rustfmt::skip]
#[derive(Debug, Clone)]
pub enum Instruction {
    Nop,
    Brk,
    Hlt,
    Err,

    Add    { d: Register, l: Register, r: AluRhs },
    AddC   { d: Register, l: Register, r: AluRhs },
    Sub    { d: Register, l: Register, r: AluRhs },
    SubB   { d: Register, l: Register, r: AluRhs },
    And    { d: Register, l: Register, r: AluRhs },
    Or     { d: Register, l: Register, r: AluRhs },
    Xor    { d: Register, l: Register, r: AluRhs },
    Shl    { d: Register, l: Register, r: AluRhs },
    Lsr    { d: Register, l: Register, r: AluRhs },
    Asr    { d: Register, l: Register, r: AluRhs },
    Mul    { d: Register, l: Register, r: AluRhs },
    MulHuu { d: Register, l: Register, r: AluRhs },
    MulHss { d: Register, l: Register, r: AluRhs },
    MulHsu { d: Register, l: Register, r: AluRhs },
    CSub   { d: Register, l: Register, r: AluRhs },
    Slc    { d: Register, s: Register },

    Ld    { d: Register, s: Register, o: Expression },
    Ld8   { d: Register, s: Register, o: Expression },
    Ld8s  { d: Register, s: Register, o: Expression },
    Ld16  { d: Register, s: Register, o: Expression },
    Ld16s { d: Register, s: Register, o: Expression },
    In    { d: Register, s: Register, o: Expression },

    St   { d: Register, o: Expression, s: Register },
    St8  { d: Register, o: Expression, s: Register },
    St16 { d: Register, o: Expression, s: Register },
    Out  { d: Register, o: Expression, s: Register },

    Jmp  { s: Register, o: Expression, indirect: bool },
    Link { d: Register, o: Expression },

    BrC   { d: Expression },
    BrZ   { d: Expression },
    BrS   { d: Expression },
    BrO   { d: Expression },
    BrNc  { d: Expression },
    BrNz  { d: Expression },
    BrNs  { d: Expression },
    BrNo  { d: Expression },
    BrULe { d: Expression },
    BrUG  { d: Expression },
    BrSL  { d: Expression },
    BrSGe { d: Expression },
    BrSLe { d: Expression },
    BrSG  { d: Expression },
    Bra   { d: Expression },

    LdUi    { d: Register, ui: Expression },
    AddPcUi { d: Register, ui: Expression },

    Sys,
    ClrK,
}
impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nop => write!(f, "nop"),
            Self::Brk => write!(f, "brk"),
            Self::Hlt => write!(f, "hlt"),
            Self::Err => write!(f, "err"),
            Self::Add { d, l, r } => write!(f, "add {}, {}, {}", d, l, r),
            Self::AddC { d, l, r } => write!(f, "addc {}, {}, {}", d, l, r),
            Self::Sub { d, l, r } => write!(f, "sub {}, {}, {}", d, l, r),
            Self::SubB { d, l, r } => write!(f, "subb {}, {}, {}", d, l, r),
            Self::And { d, l, r } => write!(f, "and {}, {}, {}", d, l, r),
            Self::Or { d, l, r } => write!(f, "or {}, {}, {}", d, l, r),
            Self::Xor { d, l, r } => write!(f, "xor {}, {}, {}", d, l, r),
            Self::Shl { d, l, r } => write!(f, "shl {}, {}, {}", d, l, r),
            Self::Lsr { d, l, r } => write!(f, "lsr {}, {}, {}", d, l, r),
            Self::Asr { d, l, r } => write!(f, "asr {}, {}, {}", d, l, r),
            Self::Mul { d, l, r } => write!(f, "mul {}, {}, {}", d, l, r),
            Self::MulHuu { d, l, r } => write!(f, "mulhuu {}, {}, {}", d, l, r),
            Self::MulHss { d, l, r } => write!(f, "mulhss {}, {}, {}", d, l, r),
            Self::MulHsu { d, l, r } => write!(f, "mulhsu {}, {}, {}", d, l, r),
            Self::CSub { d, l, r } => write!(f, "csub {}, {}, {}", d, l, r),
            Self::Slc { d, s } => write!(f, "slc {}, {}", d, s),
            Self::Ld { d, s, o } => write!(f, "ld {}, [{}, {}]", d, s, o),
            Self::Ld8 { d, s, o } => write!(f, "ld8 {}, [{}, {}]", d, s, o),
            Self::Ld8s { d, s, o } => write!(f, "ld8s {}, [{}, {}]", d, s, o),
            Self::Ld16 { d, s, o } => write!(f, "ld16 {}, [{}, {}]", d, s, o),
            Self::Ld16s { d, s, o } => write!(f, "ld16s {}, [{}, {}]", d, s, o),
            Self::In { d, s, o } => write!(f, "in {}, [{}, {}]", d, s, o),
            Self::St { d, o, s } => write!(f, "st [{}, {}], {}", d, o, s),
            Self::St8 { d, o, s } => write!(f, "st8 [{}, {}], {}", d, o, s),
            Self::St16 { d, o, s } => write!(f, "st16 [{}, {}], {}", d, o, s),
            Self::Out { d, o, s } => write!(f, "out [{}, {}], {}", d, o, s),
            Self::Jmp { s, o, indirect } => {
                if *indirect {
                    write!(f, "jmp [{}, {}]", s, o)
                } else {
                    write!(f, "jmp {}, {}", s, o)
                }
            }
            Self::Link { d, o } => write!(f, "link {}, {}", d, o),
            Self::BrC { d } => write!(f, "br.c {}", d),
            Self::BrZ { d } => write!(f, "br.z {}", d),
            Self::BrS { d } => write!(f, "br.s {}", d),
            Self::BrO { d } => write!(f, "br.o {}", d),
            Self::BrNc { d } => write!(f, "br.nc {}", d),
            Self::BrNz { d } => write!(f, "br.nz {}", d),
            Self::BrNs { d } => write!(f, "br.ns {}", d),
            Self::BrNo { d } => write!(f, "br.no {}", d),
            Self::BrULe { d } => write!(f, "br.u.le {}", d),
            Self::BrUG { d } => write!(f, "br.u.g {}", d),
            Self::BrSL { d } => write!(f, "br.s.l {}", d),
            Self::BrSGe { d } => write!(f, "br.s.ge {}", d),
            Self::BrSLe { d } => write!(f, "br.s.le {}", d),
            Self::BrSG { d } => write!(f, "br.s.g {}", d),
            Self::Bra { d } => write!(f, "bra {}", d),
            Self::LdUi { d, ui } => write!(f, "ldui {}, {}", d, ui),
            Self::AddPcUi { d, ui } => write!(f, "addpcui {}, {}", d, ui),
            Self::Sys => write!(f, "sys"),
            Self::ClrK => write!(f, "clrk"),
        }
    }
}

macro_rules! misc_inst {
    ($name:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            parser!(keyword(Keyword::$inst), =>, Instruction::$inst)(input)
        }
    };
}

misc_inst!(nop, Nop);
misc_inst!(brk, Brk);
misc_inst!(hlt, Hlt);
misc_inst!(err, Err);
misc_inst!(sys, Sys);
misc_inst!(clrk, ClrK);

fn inst_req_ws(input: TokenInput) -> ParseResult<()> {
    require(
        whitespace1,
        error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ),
    )(input)
}

fn inst_req_reg(input: TokenInput) -> ParseResult<Register> {
    require(register, error!("expected register"))(input)
}

fn inst_req_expr(input: TokenInput) -> ParseResult<Expression> {
    require(expr, error!("expected expression"))(input)
}

fn inst_req_comma(input: TokenInput) -> ParseResult<()> {
    require(comma_sep, error!("expected `,`"))(input)
}

fn inst_req_alu_rhs(input: TokenInput) -> ParseResult<AluRhs> {
    require(alu_rhs, error!("expected register or expression"))(input)
}

fn alu3_args(input: TokenInput) -> ParseResult<(Register, Register, AluRhs)> {
    map(
        parser!(inst_req_reg, <<, inst_req_comma, &, inst_req_reg, <<, inst_req_comma, &, inst_req_alu_rhs),
        |((d, l), r)| (d, l, r),
    )(input)
}

macro_rules! alu3_inst {
    ($name:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(parser!(keyword(Keyword::$inst), >>, inst_req_ws, >>, alu3_args), |(d, l, r)| Instruction::$inst { d, l, r })(input)
        }
    };
}

alu3_inst!(add, Add);
alu3_inst!(addc, AddC);
alu3_inst!(sub, Sub);
alu3_inst!(subb, SubB);
alu3_inst!(and, And);
alu3_inst!(or, Or);
alu3_inst!(xor, Xor);
alu3_inst!(shl, Shl);
alu3_inst!(lsr, Lsr);
alu3_inst!(asr, Asr);
alu3_inst!(mul, Mul);
alu3_inst!(mulhuu, MulHuu);
alu3_inst!(mulhss, MulHss);
alu3_inst!(mulhsu, MulHsu);
alu3_inst!(csub, CSub);

fn alu2_args(input: TokenInput) -> ParseResult<(Register, Register)> {
    parser!(inst_req_reg, <<, inst_req_comma, &, inst_req_reg)(input)
}

fn alu2_imm_args(input: TokenInput) -> ParseResult<(Register, Expression)> {
    parser!(inst_req_reg, <<, inst_req_comma, &, inst_req_expr)(input)
}

macro_rules! alu2_inst {
    ($name:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(parser!(keyword(Keyword::$inst), >>, inst_req_ws, >>, alu2_args)
            , |(d, s)| Instruction::$inst { d, s })(input)
        }
    };
}

alu2_inst!(slc, Slc);

fn mov(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Mov), >>, inst_req_ws, >>, alu2_args),
        |(d, s)| Instruction::Or {
            d,
            l: Register::ZERO,
            r: AluRhs::Register(s),
        },
    )(input)
}

fn ldi(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::LdI), >>, inst_req_ws, >>, alu2_imm_args),
        |(d, imm)| Instruction::Or {
            d,
            l: Register::ZERO,
            r: AluRhs::Immediate(imm),
        },
    )(input)
}

fn alu_no_store_args(input: TokenInput) -> ParseResult<(Register, AluRhs)> {
    parser!(inst_req_reg, <<, inst_req_comma, &, inst_req_alu_rhs)(input)
}

macro_rules! alu_no_store_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(parser!(keyword(Keyword::$kw), >>, inst_req_ws, >>, alu_no_store_args)
            , |(l, r)| Instruction::$inst {
                d: Register::ZERO,
                l,
                r,
            })(input)
        }
    };
}

alu_no_store_inst!(cmp, Cmp, Sub);
alu_no_store_inst!(bit, Bit, And);

fn test(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Test), >>, inst_req_ws, >>, inst_req_reg),
        |s| Instruction::Or {
            d: Register::ZERO,
            l: s,
            r: AluRhs::Register(Register::ZERO),
        },
    )(input)
}

fn inc(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Inc), >>, inst_req_ws, >>, inst_req_reg),
        |d| Instruction::Add {
            d,
            l: d,
            r: AluRhs::Immediate(Expression::new_dummy_constant(1)),
        },
    )(input)
}

fn incc(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::IncC), >>, inst_req_ws, >>, inst_req_reg),
        |d| Instruction::AddC {
            d,
            l: d,
            r: AluRhs::Immediate(Expression::new_dummy_constant(0)),
        },
    )(input)
}

fn dec(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Dec), >>, inst_req_ws, >>, inst_req_reg),
        |d| Instruction::Sub {
            d,
            l: d,
            r: AluRhs::Immediate(Expression::new_dummy_constant(1)),
        },
    )(input)
}

fn decb(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::DecB), >>, inst_req_ws, >>, inst_req_reg),
        |d| Instruction::SubB {
            d,
            l: d,
            r: AluRhs::Immediate(Expression::new_dummy_constant(0)),
        },
    )(input)
}

fn neg(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Neg), >>, inst_req_ws, >>, alu2_args),
        |(d, s)| Instruction::Sub {
            d,
            l: Register::ZERO,
            r: AluRhs::Register(s),
        },
    )(input)
}

fn negb(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::NegB), >>, inst_req_ws, >>, alu2_args),
        |(d, s)| Instruction::SubB {
            d,
            l: Register::ZERO,
            r: AluRhs::Register(s),
        },
    )(input)
}

fn not(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Not), >>, inst_req_ws, >>, alu2_args),
        |(d, s)| Instruction::Xor {
            d,
            l: s,
            r: AluRhs::Immediate(Expression::new_dummy_constant(-1)),
        },
    )(input)
}

fn offset_arg(input: TokenInput) -> ParseResult<(Register, Expression)> {
    let reg_arg = parser!(register, &, or_return(parser!(comma_sep, >>, expr), Expression::new_dummy_constant(0)));
    parser!(reg_arg, |, map(expr, |o| (Register(u5!(0)), o)))(input)
}

fn mem_arg(input: TokenInput) -> ParseResult<(Register, Expression)> {
    in_brackets(require(offset_arg, error!("expected offset")))(input)
}

fn ld_args(input: TokenInput) -> ParseResult<(Register, Register, Expression)> {
    map(
        parser!(inst_req_reg, <<, inst_req_comma, &, mem_arg),
        |(d, (s, o))| (d, s, o),
    )(input)
}

fn st_args(input: TokenInput) -> ParseResult<(Register, Expression, Register)> {
    map(
        parser!(mem_arg, <<, inst_req_comma, &, inst_req_reg),
        |((d, o), s)| (d, o, s),
    )(input)
}

macro_rules! ld_inst {
    ($name:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(parser!(keyword(Keyword::$inst), >>, inst_req_ws, >>, ld_args)
            , |(d, s, o)| Instruction::$inst { d, s, o })(input)
        }
    };
}

macro_rules! st_inst {
    ($name:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(parser!(keyword(Keyword::$inst), >>, inst_req_ws, >>, st_args)
            , |(d, o, s)| Instruction::$inst { d, o, s })(input)
        }
    };
}

ld_inst!(ld, Ld);
ld_inst!(ld8, Ld8);
ld_inst!(ld8s, Ld8s);
ld_inst!(ld16, Ld16);
ld_inst!(ld16s, Ld16s);
ld_inst!(io_in, In);

st_inst!(st, St);
st_inst!(st8, St8);
st_inst!(st16, St16);
st_inst!(io_out, Out);

fn jmp(input: TokenInput) -> ParseResult<Instruction> {
    let arg = parser!(map(mem_arg, |(s, o)| Instruction::Jmp {
        s,
        o,
        indirect: true,
    }) ,|, map(require(offset_arg, error!("expected offset")), |(s, o)| Instruction::Jmp {
            s,
            o,
            indirect: false,
        }));

    parser!(keyword(Keyword::Jmp), >>, inst_req_ws, >>, arg)(input)
}

fn link(input: TokenInput) -> ParseResult<Instruction> {
    map(
        parser!(keyword(Keyword::Link), >>, inst_req_ws, >>, inst_req_reg, <<, inst_req_comma, &, inst_req_expr),
        |(d, o)| Instruction::Link { d, o },
    )(input)
}

macro_rules! br_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(parser!(keyword(Keyword::$kw), >>, inst_req_ws, >>, inst_req_expr)
            , |d| Instruction::$inst { d })(input)
        }
    };
}

br_inst!(brc, BrC, BrC);
br_inst!(brz, BrZ, BrZ);
br_inst!(brs, BrS, BrS);
br_inst!(bro, BrO, BrO);
br_inst!(brnc, BrNc, BrNc);
br_inst!(brnz, BrNz, BrNz);
br_inst!(brns, BrNs, BrNs);
br_inst!(brno, BrNo, BrNo);
br_inst!(breq, BrEq, BrZ);
br_inst!(brneq, BrNeq, BrNz);
br_inst!(brul, BrUL, BrNc);
br_inst!(bruge, BrUGe, BrC);
br_inst!(brule, BrULe, BrULe);
br_inst!(brug, BrUG, BrUG);
br_inst!(brsl, BrSL, BrSL);
br_inst!(brsge, BrSGe, BrSGe);
br_inst!(brsle, BrSLe, BrSLe);
br_inst!(brsg, BrSG, BrSG);
br_inst!(bra, Bra, Bra);

macro_rules! ui_inst {
    ($name:ident, $inst:ident) => {
        fn $name(input: TokenInput) -> ParseResult<Instruction> {
            map(
                parser!(keyword(Keyword::$inst), >>, inst_req_ws, >>, inst_req_reg, <<, inst_req_comma, &, inst_req_expr),
                |(d, ui)| Instruction::$inst { d, ui }
            )(input)
        }
    };
}

ui_inst!(ldui, LdUi);
ui_inst!(addpcui, AddPcUi);

fn inst(input: TokenInput) -> ParseResult<Instruction> {
    choice!(
        nop, brk, hlt, err, sys, clrk, add, addc, sub, subb, and, or, xor, shl, lsr, asr, mul,
        mulhuu, mulhss, mulhsu, csub, slc, mov, ldi, cmp, bit, test, inc, incc, dec, decb, neg,
        negb, not, ld, ld8, ld8s, ld16, ld16s, io_in, st, st8, st16, io_out, jmp, link, brc, brz,
        brs, bro, brnc, brnz, brns, brno, breq, brneq, brul, bruge, brule, brug, brsl, brsge,
        brsle, brsg, bra, ldui, addpcui
    )(input)
}

#[derive(Debug, Clone)]
pub enum LineKind {
    Label(SharedString),
    Define(SharedString, Expression),
    Directive(AssemblerDirective),
    Instruction(Instruction),
}
impl Display for LineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LineKind::Label(name) => write!(f, "{}:", name),
            LineKind::Define(name, expr) => write!(f, "${} = {}", name, expr),
            LineKind::Directive(dir) => write!(f, "{}", dir),
            LineKind::Instruction(inst) => write!(f, "{}", inst),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Line {
    kind: LineKind,
    number: usize,
    span: TextSpan,
}
impl Line {
    #[inline]
    pub fn kind(&self) -> &LineKind {
        &self.kind
    }

    #[inline]
    pub fn number(&self) -> usize {
        self.number
    }

    #[inline]
    pub fn span(&self) -> &TextSpan {
        &self.span
    }
}

pub fn parse_line(line: &[Token]) -> Result<Line, ParseError> {
    let input = TokenInput::new(line);

    let line_content = choice!(
        parser!(label_def, ->, LineKind::Label),
        parser!(const_def, ->, |(name, expr)| LineKind::Define(name, expr)),
        parser!(dir, ->, LineKind::Directive),
        parser!(inst, ->, LineKind::Instruction)
    );

    let parser = parser!(whitespace0, >>, line_content, <<, whitespace0, <<, require(eof,
        error!(all "unexpected line continuation", all "this already forms a complete instruction"),
    ));

    match parser(input) {
        ParseResult::Match { value: kind, .. } => Ok(Line {
            kind,
            number: line.first().unwrap().span().line(),
            span: line
                .first()
                .unwrap()
                .span()
                .combine(line.last().unwrap().span()),
        }),
        ParseResult::NoMatch => {
            let full_span = line
                .first()
                .unwrap()
                .span()
                .combine(line.last().unwrap().span());
            Err(ParseError::new(full_span, "invalid instruction"))
        }
        ParseResult::Err(err) => Err(err),
    }
}
