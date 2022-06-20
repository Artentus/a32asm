use crate::lexer::*;
use crate::{Message, MessageKind, Register, SharedString};
use rparse::*;
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
    fn peek_kind(&self) -> Option<&'a TokenKind> {
        self.tokens.get(self.position).map(|token| token.kind())
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

type TokenParser<'a, T> = Parser<'a, T, ParseError, TokenInput<'a>>;

fn whitespace<'a>() -> TokenParser<'a, ()> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Whitespace) = input.peek_kind() {
            ParseResult::Match(input.advance(), ())
        } else {
            ParseResult::NoMatch
        }
    })
}

fn comment<'a>() -> TokenParser<'a, ()> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Comment { .. }) = input.peek_kind() {
            ParseResult::Match(input.advance(), ())
        } else {
            ParseResult::NoMatch
        }
    })
}

fn operator<'a>(op: Operator) -> TokenParser<'a, ()> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Operator(found_op)) = input.peek_kind() {
            if *found_op == op {
                ParseResult::Match(input.advance(), ())
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn operator_span<'a>(op: Operator) -> TokenParser<'a, TextSpan> {
    parser!(input: TokenInput<'a> => {
        if let Some(token) = input.peek() {
            if token.kind() == &TokenKind::Operator(op) {
                ParseResult::Match(input.advance(), token.span().clone())
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn integer_literal<'a>() -> TokenParser<'a, i64> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::IntegerLiteral(val)) = input.peek_kind() {
            ParseResult::Match(input.advance(), *val)
        } else {
            ParseResult::NoMatch
        }
    })
}

fn integer_literal_span<'a>() -> TokenParser<'a, (i64, TextSpan)> {
    parser!(input: TokenInput<'a> => {
        if let Some(token) = input.peek() {
            if let TokenKind::IntegerLiteral(val) = token.kind() {
                ParseResult::Match(input.advance(), (*val, token.span().clone()))
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

//fn char_literal<'a>() -> TokenParser<'a, char> {
//    parser!(input: TokenInput<'a> => {
//        if let Some(TokenKind::CharLiteral(c)) = input.peek_kind() {
//            ParseResult::Match(input.advance(), *c)
//        } else {
//            ParseResult::NoMatch
//        }
//    })
//}

fn char_literal_span<'a>() -> TokenParser<'a, (char, TextSpan)> {
    parser!(input: TokenInput<'a> => {
        if let Some(token) = input.peek() {
            if let TokenKind::CharLiteral(c) = token.kind() {
                ParseResult::Match(input.advance(), (*c, token.span().clone()))
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn string_literal<'a>() -> TokenParser<'a, SharedString> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::StringLiteral(s)) = input.peek_kind() {
            ParseResult::Match(input.advance(), Rc::clone(s))
        } else {
            ParseResult::NoMatch
        }
    })
}

fn directive<'a>(dir: Directive) -> TokenParser<'a, ()> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Directive(found_dir)) = input.peek_kind() {
            if *found_dir == dir {
                ParseResult::Match(input.advance(), ())
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn register<'a>() -> TokenParser<'a, Register> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Register(reg)) = input.peek_kind() {
            ParseResult::Match(input.advance(), *reg)
        } else {
            ParseResult::NoMatch
        }
    })
}

fn keyword<'a>(kw: Keyword) -> TokenParser<'a, ()> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Keyword(found_kw)) = input.peek_kind() {
            if *found_kw == kw {
                ParseResult::Match(input.advance(), ())
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn identifier<'a>() -> TokenParser<'a, SharedString> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Identifier(ident)) = input.peek_kind() {
            ParseResult::Match(input.advance(), Rc::clone(ident))
        } else {
            ParseResult::NoMatch
        }
    })
}

fn identifier_span<'a>() -> TokenParser<'a, (SharedString, TextSpan)> {
    parser!(input: TokenInput<'a> => {
        if let Some(token) = input.peek() {
            if let TokenKind::Identifier(ident) = token.kind() {
                ParseResult::Match(input.advance(), (Rc::clone(ident), token.span().clone()))
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn eof<'a>() -> TokenParser<'a, ()> {
    parser!(input: TokenInput<'a> => {
        if let Some(_) = input.peek() {
            ParseResult::NoMatch
        } else {
            ParseResult::Match(input, ())
        }
    })
}

fn whitespace0<'a>() -> TokenParser<'a, ()> {
    (whitespace() | comment()).many().map_to(())
}

fn whitespace1<'a>() -> TokenParser<'a, ()> {
    (whitespace() | comment()).many1().map_to(())
}

fn in_brackets<'a, T: 'a>(parser: TokenParser<'a, T>) -> TokenParser<'a, T> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Operator(Operator::OpenBracket)) = input.peek_kind() {
            let hint_span = input.error_span(false);
            let remaining = input.advance();

            let full_parser = whitespace0()
                >> parser.clone()
                << whitespace0()
                << operator(Operator::CloseBracket).require(|input| ParseError::new_with_hint(
                    input.error_span(false),
                    "missing closing bracket",
                    hint_span.clone(),
                    "matching open bracket here",
                ));

            full_parser.run(remaining)
        } else {
            ParseResult::NoMatch
        }
    })
}

fn label_ident<'a>() -> TokenParser<'a, (SharedString, TextSpan)> {
    identifier_span()
}

fn const_ident<'a>() -> TokenParser<'a, SharedString> {
    operator(Operator::Define)
        >> identifier().require(error!(
            "expected identifier",
            "`$` indices a constant identifier"
        ))
}

fn const_ident_span<'a>() -> TokenParser<'a, (SharedString, TextSpan)> {
    (operator_span(Operator::Define)
        & identifier_span().require(error!(
            "expected identifier",
            "`$` indices a constant identifier"
        )))
    .map(|(s1, (ident, s2))| (ident, s1.combine(&s2)))
}

fn comma_sep<'a>() -> TokenParser<'a, ()> {
    whitespace0() >> operator(Operator::Comma) << whitespace0()
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

    fn new_integer(output: (i64, TextSpan)) -> Self {
        Self {
            kind: ExpressionKind::IntegerConstant(output.0),
            span: Some(output.1),
        }
    }

    fn new_char(output: (char, TextSpan)) -> Self {
        Self {
            kind: ExpressionKind::CharConstant(output.0),
            span: Some(output.1),
        }
    }

    fn new_label(output: (SharedString, TextSpan)) -> Self {
        Self {
            kind: ExpressionKind::Label(output.0),
            span: Some(output.1),
        }
    }

    fn new_define(output: (SharedString, TextSpan)) -> Self {
        Self {
            kind: ExpressionKind::Define(output.0),
            span: Some(output.1),
        }
    }

    fn new_unary(output: ((UnaryOperator, TextSpan), Expression)) -> Self {
        let sub_span = output.1.span().unwrap().clone();
        Self {
            kind: ExpressionKind::UnaryOperator(output.0 .0, Box::new(output.1)),
            span: Some(output.0 .1.combine(&sub_span)),
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

fn leaf_expr<'a>() -> TokenParser<'a, Expression> {
    paren_expr()
        | integer_literal_span().map(Expression::new_integer)
        | char_literal_span().map(Expression::new_char)
        | label_ident().map(Expression::new_label)
        | const_ident_span().map(Expression::new_define)
}

fn unary_op<'a>() -> TokenParser<'a, (UnaryOperator, TextSpan)> {
    operator_span(Operator::Plus).map(|span| (UnaryOperator::Positive, span))
        | operator_span(Operator::Minus).map(|span| (UnaryOperator::Negative, span))
        | operator_span(Operator::Not).map(|span| (UnaryOperator::Not, span))
}

fn unary_expr<'a>() -> TokenParser<'a, Expression> {
    leaf_expr()
        | (unary_op() << whitespace0() & leaf_expr().require(error!("expected expression")))
            .map(Expression::new_unary)
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

fn mul_div_op<'a>() -> TokenParser<'a, BinaryOperator> {
    operator(Operator::Times).map_to(BinaryOperator::Multiply)
        | operator(Operator::Divide).map_to(BinaryOperator::Divide)
        | operator(Operator::Remainder).map_to(BinaryOperator::Remainder)
}

fn mul_div_expr<'a>() -> TokenParser<'a, Expression> {
    (unary_expr()
        & (whitespace0() >> mul_div_op()
            & whitespace0() >> unary_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn add_sub_op<'a>() -> TokenParser<'a, BinaryOperator> {
    operator(Operator::Plus).map_to(BinaryOperator::Add)
        | operator(Operator::Minus).map_to(BinaryOperator::Subtract)
}

fn add_sub_expr<'a>() -> TokenParser<'a, Expression> {
    (mul_div_expr()
        & (whitespace0() >> add_sub_op()
            & whitespace0() >> mul_div_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn shift_op<'a>() -> TokenParser<'a, BinaryOperator> {
    operator(Operator::ShiftLeft).map_to(BinaryOperator::ShiftLeft)
        | operator(Operator::ShiftRight).map_to(BinaryOperator::ShiftRight)
        | operator(Operator::ShiftRightArithmetic).map_to(BinaryOperator::ShiftRightArithmetic)
}

fn shift_expr<'a>() -> TokenParser<'a, Expression> {
    (add_sub_expr()
        & (whitespace0() >> shift_op()
            & whitespace0() >> add_sub_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn comp_op<'a>() -> TokenParser<'a, BinaryOperator> {
    operator(Operator::LessThan).map_to(BinaryOperator::Less)
        | operator(Operator::LessThanEquals).map_to(BinaryOperator::LessEqual)
        | operator(Operator::GreaterThan).map_to(BinaryOperator::Greater)
        | operator(Operator::GreaterThanEquals).map_to(BinaryOperator::GreaterEqual)
}

fn comp_expr<'a>() -> TokenParser<'a, Expression> {
    (shift_expr()
        & (whitespace0() >> comp_op()
            & whitespace0() >> shift_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn eq_op<'a>() -> TokenParser<'a, BinaryOperator> {
    operator(Operator::Equals).map_to(BinaryOperator::Equals)
        | operator(Operator::NotEquals).map_to(BinaryOperator::NotEquals)
}

fn eq_expr<'a>() -> TokenParser<'a, Expression> {
    (comp_expr()
        & (whitespace0() >> eq_op()
            & whitespace0() >> comp_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn and_expr<'a>() -> TokenParser<'a, Expression> {
    (eq_expr()
        & (whitespace0() >> operator(Operator::And).map_to(BinaryOperator::And)
            & whitespace0() >> eq_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn xor_expr<'a>() -> TokenParser<'a, Expression> {
    (and_expr()
        & (whitespace0() >> operator(Operator::Xor).map_to(BinaryOperator::Xor)
            & whitespace0() >> and_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn or_expr<'a>() -> TokenParser<'a, Expression> {
    (xor_expr()
        & (whitespace0() >> operator(Operator::Or).map_to(BinaryOperator::Or)
            & whitespace0() >> xor_expr().require(error!("expected expression")))
        .many())
    .map(aggregate_exprs)
}

fn paren_expr<'a>() -> TokenParser<'a, Expression> {
    parser!(input: TokenInput<'a> => {
        if let Some(TokenKind::Operator(Operator::OpenParen)) = input.peek_kind() {
            let hint_span = input.error_span(false);
            let remaining = input.advance();

            let parser = whitespace0()
                >> or_expr().require(error!("expected expression"))
                << whitespace0()
                << operator(Operator::CloseParen).require(|input| ParseError::new_with_hint(
                    input.error_span(false),
                    "missing closing parenthesis",
                    hint_span.clone(),
                    "matching open paranthesis here",
                ));

            parser.run(remaining).map(Expression::new_parenthesized)
        } else {
            ParseResult::NoMatch
        }
    })
}

fn expr<'a>() -> TokenParser<'a, Expression> {
    or_expr()
}

fn label_def<'a>() -> TokenParser<'a, SharedString> {
    identifier()
        << (whitespace0() & operator(Operator::Colon))
            .require(error!("expected `:`", "label declarations require a colon"))
}

fn const_def<'a>() -> TokenParser<'a, (SharedString, Expression)> {
    const_ident()
        & (whitespace0()
            >> operator(Operator::Assign).require(error!("expected assignment"))
            >> whitespace0()
            >> expr().require(error!("expected expression")))
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

fn inc_dir<'a>() -> TokenParser<'a, AssemblerDirective> {
    (directive(Directive::Include)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the directive and the file path"
        ))
        >> string_literal().require(error!("expected file path")))
    .map(|s| AssemblerDirective::Include(s))
}

fn addr_dir<'a>() -> TokenParser<'a, AssemblerDirective> {
    let addr_literal = integer_literal().verify(|addr| *addr <= (u32::MAX as i64));
    (directive(Directive::Address)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the directive and the address"
        ))
        >> addr_literal.require(error!("expected address")))
    .map(|addr| AssemblerDirective::Address(addr as u32))
}

fn align_dir<'a>() -> TokenParser<'a, AssemblerDirective> {
    let align_literal = integer_literal().verify(|align| *align <= (u32::MAX as i64));
    (directive(Directive::Align)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the directive and the alignment"
        ))
        >> align_literal.require(error!("expected alignment")))
    .map(|align| AssemblerDirective::Align(align as u32))
}

macro_rules! int_dir {
    ($name:ident, $dir:ident) => {
        fn $name<'a>() -> TokenParser<'a, AssemblerDirective> {
            (directive(Directive::$dir)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the directive and the data"
                ))
                >> expr()
                    .sep_by1(comma_sep(), true)
                    .require(error!("expected data")))
            .map(AssemblerDirective::$dir)
        }
    };
}

int_dir!(int8_dir, Int8);
int_dir!(int16_dir, Int16);
int_dir!(int32_dir, Int32);
int_dir!(int64_dir, Int64);

macro_rules! string_dir {
    ($name:ident, $dir:ident) => {
        fn $name<'a>() -> TokenParser<'a, AssemblerDirective> {
            (directive(Directive::$dir)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the directive and the string"
                ))
                >> string_literal().require(error!("expected string")))
            .map(AssemblerDirective::$dir)
        }
    };
}

string_dir!(ascii_dir, Ascii);
string_dir!(asciiz_dir, AsciiZ);
string_dir!(utf8_dir, Utf8);
string_dir!(utf16_dir, Utf16);
string_dir!(unicode_dir, Unicode);

fn dir<'a>() -> TokenParser<'a, AssemblerDirective> {
    inc_dir()
        | addr_dir()
        | align_dir()
        | int8_dir()
        | int16_dir()
        | int32_dir()
        | int64_dir()
        | ascii_dir()
        | asciiz_dir()
        | utf8_dir()
        | utf16_dir()
        | unicode_dir()
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

fn alu_rhs<'a>() -> TokenParser<'a, AluRhs> {
    register().map(AluRhs::Register) | expr().map(AluRhs::Immediate)
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

    Jmp { s: Register, o: Expression, indirect: bool },

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
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            keyword(Keyword::$inst).map_to(Instruction::$inst)
        }
    };
}

misc_inst!(nop, Nop);
misc_inst!(brk, Brk);
misc_inst!(hlt, Hlt);
misc_inst!(err, Err);
misc_inst!(sys, Sys);
misc_inst!(clrk, ClrK);

fn alu3_args<'a>() -> TokenParser<'a, (Register, Register, AluRhs)> {
    (register().require(error!("expected register")) << comma_sep().require(error!("expected `,`"))
        & register().require(error!("expected register"))
            << comma_sep().require(error!("expected `,`"))
        & alu_rhs().require(error!("expected register or expression")))
    .map(|((d, l), r)| (d, l, r))
}

macro_rules! alu3_inst {
    ($name:ident, $inst:ident) => {
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$inst)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> alu3_args())
            .map(|(d, l, r)| Instruction::$inst { d, l, r })
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

fn alu2_args<'a>() -> TokenParser<'a, (Register, Register)> {
    register().require(error!("expected register")) << comma_sep().require(error!("expected `,`"))
        & register().require(error!("expected register"))
}

fn alu2_imm_args<'a>() -> TokenParser<'a, (Register, Expression)> {
    register().require(error!("expected register")) << comma_sep().require(error!("expected `,`"))
        & expr().require(error!("expected expression"))
}

macro_rules! alu2_inst {
    ($name:ident, $inst:ident) => {
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$inst)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> alu2_args())
            .map(|(d, s)| Instruction::$inst { d, s })
        }
    };
}

alu2_inst!(slc, Slc);

fn mov<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::Mov)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> alu2_args())
    .map(|(d, s)| Instruction::Or {
        d,
        l: Register::ZERO,
        r: AluRhs::Register(s),
    })
}

fn ldi<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::LdI)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> alu2_imm_args())
    .map(|(d, imm)| Instruction::Or {
        d,
        l: Register::ZERO,
        r: AluRhs::Immediate(imm),
    })
}

fn alu_no_store_args<'a>() -> TokenParser<'a, (Register, AluRhs)> {
    register().require(error!("expected register")) << comma_sep().require(error!("expected `,`"))
        & alu_rhs().require(error!("expected register or expression"))
}

macro_rules! alu_no_store_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$kw)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> alu_no_store_args())
            .map(|(l, r)| Instruction::$inst {
                d: Register::ZERO,
                l,
                r,
            })
        }
    };
}

alu_no_store_inst!(cmp, Cmp, Sub);
alu_no_store_inst!(bit, Bit, And);

fn test<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::Test)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> register().require(error!("expected register")))
    .map(|s| Instruction::Or {
        d: Register::ZERO,
        l: s,
        r: AluRhs::Register(Register::ZERO),
    })
}

fn inc<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::Inc)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> register().require(error!("expected register")))
    .map(|d| Instruction::Add {
        d,
        l: d,
        r: AluRhs::Immediate(Expression::new_dummy_constant(1)),
    })
}

fn incc<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::IncC)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> register().require(error!("expected register")))
    .map(|d| Instruction::AddC {
        d,
        l: d,
        r: AluRhs::Immediate(Expression::new_dummy_constant(0)),
    })
}

fn dec<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::Dec)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> register().require(error!("expected register")))
    .map(|d| Instruction::Sub {
        d,
        l: d,
        r: AluRhs::Immediate(Expression::new_dummy_constant(1)),
    })
}

fn decb<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::DecB)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> register().require(error!("expected register")))
    .map(|d| Instruction::SubB {
        d,
        l: d,
        r: AluRhs::Immediate(Expression::new_dummy_constant(0)),
    })
}

fn neg<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::Neg)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> alu2_args())
    .map(|(d, s)| Instruction::Sub {
        d,
        l: Register::ZERO,
        r: AluRhs::Register(s),
    })
}

fn negb<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::NegB)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> alu2_args())
    .map(|(d, s)| Instruction::SubB {
        d,
        l: Register::ZERO,
        r: AluRhs::Register(s),
    })
}

fn not<'a>() -> TokenParser<'a, Instruction> {
    (keyword(Keyword::Not)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> alu2_args())
    .map(|(d, s)| Instruction::Xor {
        d,
        l: s,
        r: AluRhs::Immediate(Expression::new_dummy_constant(-1)),
    })
}

fn offset_arg<'a>() -> TokenParser<'a, (Register, Expression)> {
    (register()
        & (comma_sep() >> expr())
            .opt()
            .map(|o| o.unwrap_or(Expression::new_dummy_constant(0))))
        | expr().map(|o| (Register(u5!(0)), o))
}

fn mem_arg<'a>() -> TokenParser<'a, (Register, Expression)> {
    in_brackets(offset_arg().require(error!("expected offset")))
}

fn ld_args<'a>() -> TokenParser<'a, (Register, Register, Expression)> {
    (register().require(error!("expected register")) << comma_sep().require(error!("expected `,`"))
        & mem_arg())
    .map(|(d, (s, o))| (d, s, o))
}

fn st_args<'a>() -> TokenParser<'a, (Register, Expression, Register)> {
    (mem_arg() << comma_sep().require(error!("expected `,`"))
        & register().require(error!("expected register")))
    .map(|((d, o), s)| (d, o, s))
}

macro_rules! ld_inst {
    ($name:ident, $inst:ident) => {
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$inst)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> ld_args())
            .map(|(d, s, o)| Instruction::$inst { d, s, o })
        }
    };
}

macro_rules! st_inst {
    ($name:ident, $inst:ident) => {
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$inst)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> st_args())
            .map(|(d, o, s)| Instruction::$inst { d, o, s })
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

fn jmp<'a>() -> TokenParser<'a, Instruction> {
    keyword(Keyword::Jmp)
        >> whitespace1().require(error!(
            "expected whitespace",
            "whitespace is required between the instruction and its arguments"
        ))
        >> (mem_arg().map(|(s, o)| Instruction::Jmp {
            s,
            o,
            indirect: true,
        }) | offset_arg()
            .require(error!("expected offset"))
            .map(|(s, o)| Instruction::Jmp {
                s,
                o,
                indirect: false,
            }))
}

macro_rules! br_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$kw)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> expr().require(error!("expected expression")))
            .map(|d| Instruction::$inst { d })
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
        fn $name<'a>() -> TokenParser<'a, Instruction> {
            (keyword(Keyword::$inst)
                >> whitespace1().require(error!(
                    "expected whitespace",
                    "whitespace is required between the instruction and its arguments"
                ))
                >> register().require(error!("expected register"))
                << comma_sep().require(error!("expected `,`"))
                & expr().require(error!("expected expression")))
            .map(|(d, ui)| Instruction::$inst { d, ui })
        }
    };
}

ui_inst!(ldui, LdUi);
ui_inst!(addpcui, AddPcUi);

fn inst<'a>() -> TokenParser<'a, Instruction> {
    nop()
        | brk()
        | hlt()
        | err()
        | sys()
        | clrk()
        | add()
        | addc()
        | sub()
        | subb()
        | and()
        | or()
        | xor()
        | shl()
        | lsr()
        | asr()
        | mul()
        | mulhuu()
        | mulhss()
        | mulhsu()
        | csub()
        | slc()
        | mov()
        | ldi()
        | cmp()
        | bit()
        | test()
        | inc()
        | incc()
        | dec()
        | decb()
        | neg()
        | negb()
        | not()
        | ld()
        | ld8()
        | ld8s()
        | ld16()
        | ld16s()
        | io_in()
        | st()
        | st8()
        | st16()
        | io_out()
        | jmp()
        | brc()
        | brz()
        | brs()
        | bro()
        | brnc()
        | brnz()
        | brns()
        | brno()
        | breq()
        | brneq()
        | brul()
        | bruge()
        | brule()
        | brug()
        | brsl()
        | brsge()
        | brsle()
        | brsg()
        | bra()
        | ldui()
        | addpcui()
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

    let line_content = label_def().map(LineKind::Label)
        | const_def().map(|(name, expr)| LineKind::Define(name, expr))
        | dir().map(LineKind::Directive)
        | inst().map(LineKind::Instruction);

    let parser = whitespace0() >> line_content << whitespace0() << eof().require(
        error!(all "unexpected line continuation", all "this already forms a complete instruction"),
    );

    match parser.run(input) {
        ParseResult::Match(_, kind) => Ok(Line {
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
        ParseResult::Err(_, err) => Err(err),
    }
}
