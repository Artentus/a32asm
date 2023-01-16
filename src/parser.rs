use crate::lexer::*;
use crate::{Message, MessageKind, Register, SharedString};
use langbox::*;
use std::borrow::Cow;
use std::fmt::Display;
use std::rc::Rc;

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
        file_server: &FileServer,
    ) -> std::io::Result<()> {
        self.error_message.pretty_print(writer, file_server)?;

        if let Some(hint_message) = &self.hint_message {
            hint_message.pretty_print(writer, file_server)?;
        }

        Ok(())
    }
}

fn get_error_span<Kind>(input: TokenStream<Kind>) -> TextSpan {
    if let Some(token) = input.peek() {
        token.span
    } else {
        input.empty_span()
    }
}

fn get_hint_span<Kind>(input: TokenStream<Kind>) -> TextSpan {
    if let Some(token) = input.consumed().last() {
        token.span
    } else {
        input.empty_span()
    }
}

fn get_full_error_span<Kind>(input: TokenStream<Kind>) -> TextSpan {
    if let Some(first) = input.remaining().first() && let Some(last) = input.remaining().last() {
        first.span.join(&last.span)
    } else {
        input.empty_span()
    }
}

fn get_full_hint_span<Kind>(input: TokenStream<Kind>) -> TextSpan {
    if let Some(first) = input.consumed().first() && let Some(last) = input.consumed().last() {
        first.span.join(&last.span)
    } else {
        input.empty_span()
    }
}

macro_rules! error {
    ($error_msg:expr) => {
        |input| ParseError::new(get_error_span(input), $error_msg)
    };
    ($error_msg:expr, $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                get_error_span(input),
                $error_msg,
                get_hint_span(input),
                $hint_msg,
            )
        }
    };

    (all $error_msg:expr) => {
        |input| ParseError::new(get_full_error_span(input), $error_msg)
    };
    (all $error_msg:expr, $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                get_full_error_span(input),
                $error_msg,
                get_hint_span(input),
                $hint_msg,
            )
        }
    };
    ($error_msg:expr, all $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                get_error_span(input),
                $error_msg,
                get_full_hint_span(input),
                $hint_msg,
            )
        }
    };
    (all $error_msg:expr, all $hint_msg:expr) => {
        |input| {
            ParseError::new_with_hint(
                get_full_error_span(input),
                $error_msg,
                get_full_hint_span(input),
                $hint_msg,
            )
        }
    };
}

pub trait Parser<T> = langbox::Parser<TokenKind, T, ParseError>;

fn comment() -> impl Parser<()> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::Comment { .. } = &token.kind {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn operator(op: Operator) -> impl Parser<()> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::Operator(t_op) = &token.kind && (*t_op == op) {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn integer_literal() -> impl Parser<i64> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::IntegerLiteral(val) = &token.kind {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: *val,
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn char_literal() -> impl Parser<char> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::CharLiteral(c) = &token.kind {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: *c,
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn string_literal() -> impl Parser<SharedString> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::StringLiteral(s) = &token.kind {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: Rc::clone(s),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn directive(dir: Directive) -> impl Parser<()> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::Directive(t_dir) = &token.kind && (*t_dir == dir) {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn register() -> impl Parser<Register> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::Register(reg) = &token.kind {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: *reg,
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn keyword(kw: Keyword) -> impl Parser<()> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::Keyword(t_kw) = &token.kind && (*t_kw == kw) {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: (),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn identifier() -> impl Parser<SharedString> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() {
            if let TokenKind::Identifier(ident) = &token.kind {
                ParseResult::Match {
                    remaining: input.advance(),
                    span: token.span,
                    value: Rc::clone(ident),
                }
            } else {
                ParseResult::NoMatch
            }
        } else {
            ParseResult::NoMatch
        }
    })
}

fn verify<T>(p: impl Parser<T>, f: impl Fn(&T) -> bool + Copy) -> impl Parser<T> {
    parse_fn!(|input| match p.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => {
            if f(&value) {
                ParseResult::Match {
                    value,
                    span,
                    remaining,
                }
            } else {
                ParseResult::NoMatch
            }
        }
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

fn map_with_span<T, U>(p: impl Parser<T>, f: impl Fn(T, TextSpan) -> U + Copy) -> impl Parser<U> {
    parse_fn!(|input| match p.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => {
            ParseResult::Match {
                value: f(value, span),
                span,
                remaining,
            }
        }
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

fn in_brackets<T>(p: impl Parser<T>) -> impl Parser<T> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() && let TokenKind::Operator(Operator::OpenBracket) = &token.kind {
            let closing = parser!({operator(Operator::CloseBracket)}!![|input| {
                ParseError::new_with_hint(
                    input.empty_span(),
                    "missing closing bracket",
                    token.span,
                    "matching open bracket here",
                )
            }]);

            let remaining = input.advance();
            parser!(p <. closing).run(remaining)
        } else {
            ParseResult::NoMatch
        }
    })
}

fn label_ident() -> impl Parser<SharedString> {
    identifier()
}

fn const_ident() -> impl Parser<SharedString> {
    parser!({operator(Operator::Define)} .> {identifier()}!![error!(
        "expected identifier",
        "`$` indicates a constant identifier"
    )])
}

fn comma_sep() -> impl Parser<()> {
    operator(Operator::Comma)
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
    span: TextSpan,
}
impl Expression {
    #[inline]
    const fn new_integer(val: i64, span: TextSpan) -> Self {
        Self {
            kind: ExpressionKind::IntegerConstant(val),
            span,
        }
    }

    #[inline]
    const fn new_char(c: char, span: TextSpan) -> Self {
        Self {
            kind: ExpressionKind::CharConstant(c),
            span,
        }
    }

    #[inline]
    const fn new_label(name: SharedString, span: TextSpan) -> Self {
        Self {
            kind: ExpressionKind::Label(name),
            span,
        }
    }

    #[inline]
    const fn new_define(name: SharedString, span: TextSpan) -> Self {
        Self {
            kind: ExpressionKind::Define(name),
            span,
        }
    }

    #[inline]
    fn new_unary(output: (UnaryOperator, Expression), span: TextSpan) -> Self {
        Self {
            kind: ExpressionKind::UnaryOperator(output.0, Box::new(output.1)),
            span,
        }
    }

    #[inline]
    fn new_parenthesized(expr: Expression) -> Self {
        let span = expr.span();

        Self {
            kind: ExpressionKind::Parenthesized(Box::new(expr)),
            span,
        }
    }

    #[inline]
    pub const fn kind(&self) -> &ExpressionKind {
        &self.kind
    }

    #[inline]
    pub const fn span(&self) -> TextSpan {
        self.span
    }
}
impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.kind, f)
    }
}

fn leaf_expr() -> impl Parser<Expression> {
    let integer_literal = map_with_span(integer_literal(), Expression::new_integer);
    let char_literal = map_with_span(char_literal(), Expression::new_char);
    let label_ident = map_with_span(label_ident(), Expression::new_label);
    let const_ident = map_with_span(const_ident(), Expression::new_define);

    choice!(
        paren_expr(),
        integer_literal,
        char_literal,
        label_ident,
        const_ident,
    )
}

fn unary_op() -> impl Parser<UnaryOperator> {
    let positive = parser!({operator(Operator::Plus)}=>[UnaryOperator::Positive]);
    let negative = parser!({operator(Operator::Minus)}=>[UnaryOperator::Negative]);
    let not = parser!({operator(Operator::Not)}=>[UnaryOperator::Not]);

    choice!(positive, negative, not,)
}

fn unary_expr() -> impl Parser<Expression> {
    let unary = map_with_span(
        parser!({unary_op()} <.> {leaf_expr()}!![error!("expected expression")]),
        Expression::new_unary,
    );
    choice!(unary, leaf_expr())
}

fn aggregate_exprs<'a>(output: (Expression, Vec<(BinaryOperator, Expression)>)) -> Expression {
    let mut expr = output.0;
    for (op, rhs) in output.1 {
        let span = expr.span().join(&rhs.span());

        expr = Expression {
            kind: ExpressionKind::BinaryOperator(op, Box::new(expr), Box::new(rhs)),
            span,
        };
    }
    expr
}

fn mul_div_op() -> impl Parser<BinaryOperator> {
    let multiply = parser!({operator(Operator::Times)}=>[BinaryOperator::Multiply]);
    let divide = parser!({operator(Operator::Divide)}=>[BinaryOperator::Divide]);
    let remainder = parser!({operator(Operator::Remainder)}=>[BinaryOperator::Remainder]);

    choice!(multiply, divide, remainder,)
}

fn mul_div_expr() -> impl Parser<Expression> {
    let tail = parser!({mul_div_op()} <.> {unary_expr()}!![error!("expected expression")]);
    parser!(({unary_expr()} <.> *tail)->[aggregate_exprs])
}

fn add_sub_op() -> impl Parser<BinaryOperator> {
    let add = parser!({operator(Operator::Plus)}=>[BinaryOperator::Add]);
    let subtract = parser!({operator(Operator::Minus)}=>[BinaryOperator::Subtract]);

    choice!(add, subtract,)
}

fn add_sub_expr() -> impl Parser<Expression> {
    let tail = parser!({add_sub_op()} <.> {mul_div_expr()}!![error!("expected expression")]);
    parser!(({unary_expr()} <.> *tail)->[aggregate_exprs])
}

fn shift_op() -> impl Parser<BinaryOperator> {
    let shift_left = parser!({operator(Operator::ShiftLeft)}=>[BinaryOperator::ShiftLeft]);
    let shift_right = parser!({operator(Operator::ShiftRight)}=>[BinaryOperator::ShiftRight]);
    let shift_right_arithmetic =
        parser!({operator(Operator::ShiftRightArithmetic)}=>[BinaryOperator::ShiftRightArithmetic]);

    choice!(shift_left, shift_right, shift_right_arithmetic,)
}

fn shift_expr() -> impl Parser<Expression> {
    let tail = parser!({shift_op()} <.> {add_sub_expr()}!![error!("expected expression")]);
    parser!(({add_sub_expr()} <.> *tail)->[aggregate_exprs])
}

fn comp_op() -> impl Parser<BinaryOperator> {
    let less_than = parser!({operator(Operator::LessThan)}=>[BinaryOperator::Less]);
    let less_than_equals =
        parser!({operator(Operator::LessThanEquals)}=>[BinaryOperator::LessEqual]);
    let greater_than = parser!({operator(Operator::GreaterThan)}=>[BinaryOperator::Greater]);
    let greater_than_equals =
        parser!({operator(Operator::GreaterThanEquals)}=>[BinaryOperator::GreaterEqual]);

    choice!(
        less_than,
        less_than_equals,
        greater_than,
        greater_than_equals,
    )
}

fn comp_expr() -> impl Parser<Expression> {
    let tail = parser!({comp_op()} <.> {shift_expr()}!![error!("expected expression")]);
    parser!(({shift_expr()} <.> *tail)->[aggregate_exprs])
}

fn eq_op() -> impl Parser<BinaryOperator> {
    let equals = parser!({operator(Operator::Equals)}=>[BinaryOperator::Equals]);
    let not_equals = parser!({operator(Operator::NotEquals)}=>[BinaryOperator::NotEquals]);

    choice!(equals, not_equals,)
}

fn eq_expr() -> impl Parser<Expression> {
    let tail = parser!({eq_op()} <.> {comp_expr()}!![error!("expected expression")]);
    parser!(({comp_expr()} <.> *tail)->[aggregate_exprs])
}

fn and_expr() -> impl Parser<Expression> {
    let op = parser!({operator(Operator::And)}=>[BinaryOperator::And]);
    let tail = parser!(op <.> {eq_expr()}!![error!("expected expression")]);
    parser!(({eq_expr()} <.> *tail)->[aggregate_exprs])
}

fn xor_expr() -> impl Parser<Expression> {
    let op = parser!({operator(Operator::Xor)}=>[BinaryOperator::Xor]);
    let tail = parser!(op <.> {and_expr()}!![error!("expected expression")]);
    parser!(({and_expr()} <.> *tail)->[aggregate_exprs])
}

fn or_expr() -> impl Parser<Expression> {
    let op = parser!({operator(Operator::Or)}=>[BinaryOperator::Or]);
    let tail = parser!(op <.> {xor_expr()}!![error!("expected expression")]);
    parser!(({xor_expr()} <.> *tail)->[aggregate_exprs])
}

fn paren_expr() -> impl Parser<Expression> {
    parse_fn!(|input| {
        if let Some(token) = input.peek() && let TokenKind::Operator(Operator::OpenParen) = &token.kind {
            let closing = parser!({operator(Operator::CloseParen)}!![|input| {
                ParseError::new_with_hint(
                    input.empty_span(),
                    "missing closing parenthesis",
                    token.span,
                    "matching open paranthesis here",
                )
            }]);

            let remaining = input.advance();
            parser!(({or_expr()}!![error!("expected expression")] <. closing)->[Expression::new_parenthesized]).run(remaining)
        } else {
            ParseResult::NoMatch
        }
    })
}

fn expr() -> impl Parser<Expression> {
    or_expr()
}

fn label_def() -> impl Parser<SharedString> {
    parser!({identifier()} <. {operator(Operator::Colon)}!![error!("expected `:`", "label declarations require a colon")])
}

fn const_def() -> impl Parser<(SharedString, Expression)> {
    parser!({const_ident()} <. {operator(Operator::Assign)}!![error!("expected assignment")] <.> {expr()}!![error!("expected expression")])
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

fn inc_dir() -> impl Parser<AssemblerDirective> {
    parser!(({directive(Directive::Include)} .> {string_literal()}!![error!("expected file path")])->[|s| AssemblerDirective::Include(s)])
}

fn addr_dir() -> impl Parser<AssemblerDirective> {
    let addr = verify(integer_literal(), |addr| {
        (*addr <= (u32::MAX as i64)) & (*addr >= 0)
    });
    parser!(({directive(Directive::Address)} .> addr!![error!("expected address")])->[|addr| AssemblerDirective::Address(addr as u32)])
}

fn align_dir() -> impl Parser<AssemblerDirective> {
    let align = verify(integer_literal(), |align| {
        (*align <= (u32::MAX as i64)) & (*align >= 0)
    });
    parser!(({directive(Directive::Align)} .> align!![error!("expected address")])->[|align| AssemblerDirective::Align(align as u32)])
}

macro_rules! int_dir {
    ($name:ident, $dir:ident) => {
        fn $name() -> impl Parser<AssemblerDirective> {
            parser!(({directive(Directive::$dir)} .> {sep_by(expr(), comma_sep(), false, true)}!![error!("expected data")])->[AssemblerDirective::$dir])
        }
    };
}

int_dir!(int8_dir, Int8);
int_dir!(int16_dir, Int16);
int_dir!(int32_dir, Int32);
int_dir!(int64_dir, Int64);

macro_rules! string_dir {
    ($name:ident, $dir:ident) => {
        fn $name() -> impl Parser<AssemblerDirective> {
            parser!(({directive(Directive::$dir)} .> {string_literal()}!![error!("expected string")])->[AssemblerDirective::$dir])
        }
    };
}

string_dir!(ascii_dir, Ascii);
string_dir!(asciiz_dir, AsciiZ);
string_dir!(utf8_dir, Utf8);
string_dir!(utf16_dir, Utf16);
string_dir!(unicode_dir, Unicode);

fn dir() -> impl Parser<AssemblerDirective> {
    choice!(
        inc_dir(),
        addr_dir(),
        align_dir(),
        int8_dir(),
        int16_dir(),
        int32_dir(),
        int64_dir(),
        ascii_dir(),
        asciiz_dir(),
        utf8_dir(),
        utf16_dir(),
        unicode_dir(),
    )
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

fn alu_rhs() -> impl Parser<AluRhs> {
    parser!({register()}->[AluRhs::Register] <|> {expr()}->[AluRhs::Immediate])
}

#[rustfmt::skip]
#[derive(Debug, Clone)]
pub enum Instruction {
    Nop,
    Brk,
    Hlt,
    Err,
    Sys,
    ClrK,

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

    Jmp  { s: Register, o: Expression },
    Link { d: Register, o: Expression },

    LdUi    { d: Register, ui: Expression },
    AddPcUi { d: Register, ui: Expression },

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
    Jr    { d: Expression },

    MvC   { d: Register, l: Register, r: AluRhs },
    MvZ   { d: Register, l: Register, r: AluRhs },
    MvS   { d: Register, l: Register, r: AluRhs },
    MvO   { d: Register, l: Register, r: AluRhs },
    MvNc  { d: Register, l: Register, r: AluRhs },
    MvNz  { d: Register, l: Register, r: AluRhs },
    MvNs  { d: Register, l: Register, r: AluRhs },
    MvNo  { d: Register, l: Register, r: AluRhs },
    MvULe { d: Register, l: Register, r: AluRhs },
    MvUG  { d: Register, l: Register, r: AluRhs },
    MvSL  { d: Register, l: Register, r: AluRhs },
    MvSGe { d: Register, l: Register, r: AluRhs },
    MvSLe { d: Register, l: Register, r: AluRhs },
    MvSG  { d: Register, l: Register, r: AluRhs },
    Mov   { d: Register, s: Register },
    LdI   { d: Register, s: Expression },
}
impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nop => write!(f, "nop"),
            Self::Brk => write!(f, "brk"),
            Self::Hlt => write!(f, "hlt"),
            Self::Err => write!(f, "err"),
            Self::Sys => write!(f, "sys"),
            Self::ClrK => write!(f, "clrk"),
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
            Self::Jmp { s, o } => write!(f, "jmp {}, {}", s, o),
            Self::Link { d, o } => write!(f, "link {}, {}", d, o),
            Self::LdUi { d, ui } => write!(f, "ldui {}, {}", d, ui),
            Self::AddPcUi { d, ui } => write!(f, "addpcui {}, {}", d, ui),
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
            Self::Jr { d } => write!(f, "jr {}", d),
            Self::MvC { d, l, r } => write!(f, "mv.c {}, {}, {}", d, l, r),
            Self::MvZ { d, l, r } => write!(f, "mv.z {}, {}, {}", d, l, r),
            Self::MvS { d, l, r } => write!(f, "mv.s {}, {}, {}", d, l, r),
            Self::MvO { d, l, r } => write!(f, "mv.o {}, {}, {}", d, l, r),
            Self::MvNc { d, l, r } => write!(f, "mv.nc {}, {}, {}", d, l, r),
            Self::MvNz { d, l, r } => write!(f, "mv.nz {}, {}, {}", d, l, r),
            Self::MvNs { d, l, r } => write!(f, "mv.ns {}, {}, {}", d, l, r),
            Self::MvNo { d, l, r } => write!(f, "mv.no {}, {}, {}", d, l, r),
            Self::MvULe { d, l, r } => write!(f, "mv.u.le {}, {}, {}", d, l, r),
            Self::MvUG { d, l, r } => write!(f, "mv.u.g {}, {}, {}", d, l, r),
            Self::MvSL { d, l, r } => write!(f, "mv.s.l {}, {}, {}", d, l, r),
            Self::MvSGe { d, l, r } => write!(f, "mv.s.ge {}, {}, {}", d, l, r),
            Self::MvSLe { d, l, r } => write!(f, "mv.s.le {}, {}, {}", d, l, r),
            Self::MvSG { d, l, r } => write!(f, "mv.s.g {}, {}, {}", d, l, r),
            Self::Mov { d, s } => write!(f, "mov {}, {}", d, s),
            Self::LdI { d, s } => write!(f, "ldi {}, {}", d, s),
        }
    }
}

macro_rules! misc_inst {
    ($name:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!({keyword(Keyword::$inst)}->[|_| Instruction::$inst])
        }
    };
}

misc_inst!(nop, Nop);
misc_inst!(brk, Brk);
misc_inst!(hlt, Hlt);
misc_inst!(err, Err);
misc_inst!(sys, Sys);
misc_inst!(clrk, ClrK);

fn inst_req_reg() -> impl Parser<Register> {
    parser!({register()}!![error!("expected register")])
}

fn inst_req_expr() -> impl Parser<Expression> {
    parser!({expr()}!![error!("expected expression")])
}

fn inst_req_comma() -> impl Parser<()> {
    parser!({comma_sep()}!![error!("expected `,`")])
}

fn inst_req_alu_rhs() -> impl Parser<AluRhs> {
    parser!({alu_rhs()}!![error!("expected register or expression")])
}

fn alu_args() -> impl Parser<(Register, Register, AluRhs)> {
    parser!(({inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_alu_rhs()})->[|((d, l), r)| (d, l, r)])
}

macro_rules! alu_inst {
    ($name:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$inst)} .> {alu_args()})->[|(d, l, r)| Instruction::$inst { d, l, r }])
        }
    };
}

alu_inst!(add, Add);
alu_inst!(addc, AddC);
alu_inst!(sub, Sub);
alu_inst!(subb, SubB);
alu_inst!(and, And);
alu_inst!(or, Or);
alu_inst!(xor, Xor);
alu_inst!(shl, Shl);
alu_inst!(lsr, Lsr);
alu_inst!(asr, Asr);
alu_inst!(mul, Mul);

fn alu_no_store_args() -> impl Parser<(Register, AluRhs)> {
    parser!({inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_alu_rhs()})
}

macro_rules! alu_no_store_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$kw)} .> {alu_no_store_args()})->[|(l, r)| Instruction::$inst {
                d: Register::ZERO,
                l,
                r,
            }])
        }
    };
}

alu_no_store_inst!(cmp, Cmp, Sub);
alu_no_store_inst!(bit, Bit, And);

fn test() -> impl Parser<Instruction> {
    parser!(({keyword(Keyword::Test)} .> {inst_req_reg()})->[|s| Instruction::Or {
        d: Register::ZERO,
        l: s,
        r: AluRhs::Register(Register::ZERO),
    }])
}

macro_rules! count_inst {
    ($name:ident, $kw:ident, $inst:ident, $imm:literal) => {
        fn $name() -> impl Parser<Instruction> {
            let p = parser!({keyword(Keyword::$kw)} .> {inst_req_reg()});

            parse_fn!(|input| match p.run(input)? {
                InfallibleParseResult::Match { value, span, remaining } => {
                    ParseResult::Match {
                        value: Instruction::$inst {
                            d: value,
                            l: value,
                            r: AluRhs::Immediate(Expression::new_integer($imm, remaining.empty_span())),
                        },
                        span,
                        remaining,
                    }
                }
                InfallibleParseResult::NoMatch => ParseResult::NoMatch,
            })
        }
    };
}

count_inst!(inc, Inc, Add, 1);
count_inst!(incc, IncC, AddC, 0);
count_inst!(dec, Dec, Sub, 1);
count_inst!(decb, DecB, SubB, 0);

fn neg() -> impl Parser<Instruction> {
    parser!(({keyword(Keyword::Neg)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_reg()})->[|(d, s)| Instruction::Sub {
        d,
        l: Register::ZERO,
        r: AluRhs::Register(s),
    }])
}

fn negb() -> impl Parser<Instruction> {
    parser!(({keyword(Keyword::NegB)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_reg()})->[|(d, s)| Instruction::SubB {
        d,
        l: Register::ZERO,
        r: AluRhs::Register(s),
    }])
}

fn not() -> impl Parser<Instruction> {
    let p = parser!({keyword(Keyword::Not)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_reg()});

    parse_fn!(|input| match p.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => {
            ParseResult::Match {
                value: Instruction::Xor {
                    d: value.0,
                    l: value.1,
                    r: AluRhs::Immediate(Expression::new_integer(-1, remaining.empty_span())),
                },
                span,
                remaining,
            }
        }
        InfallibleParseResult::NoMatch => ParseResult::NoMatch,
    })
}

fn offset_arg() -> impl Parser<(Register, Expression)> {
    let comma_expr = parser!({comma_sep()} .> {expr()});
    let expr_or_zero = parse_fn!(|input| match comma_expr.run(input)? {
        InfallibleParseResult::Match {
            value,
            span,
            remaining,
        } => ParseResult::Match {
            value,
            span,
            remaining
        },
        InfallibleParseResult::NoMatch => ParseResult::Match {
            value: Expression::new_integer(0, input.empty_span()),
            span: input.empty_span(),
            remaining: input,
        },
    });

    let reg_arg = parser!({register()} <.> expr_or_zero);
    parser!(reg_arg <|> {expr()}->[|o| (Register(u5!(0)), o)])
}

fn mem_arg() -> impl Parser<(Register, Expression)> {
    in_brackets(parser!({offset_arg()}!![error!("expected offset")]))
}

fn ld_args() -> impl Parser<(Register, Register, Expression)> {
    parser!(({inst_req_reg()} <. {inst_req_comma()} <.> {mem_arg()})->[|(d, (s, o))| (d, s, o)])
}

fn st_args() -> impl Parser<(Register, Expression, Register)> {
    parser!(({mem_arg()} <. {inst_req_comma()} <.> {inst_req_reg()})->[|((d, o), s)| (d, o, s)])
}

macro_rules! ld_inst {
    ($name:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$inst)} .> {ld_args()})->[|(d, s, o)| Instruction::$inst { d, s, o }])
        }
    };
}

macro_rules! st_inst {
    ($name:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$inst)} .> {st_args()})->[|(d, o, s)| Instruction::$inst { d, o, s }])
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

fn jmp() -> impl Parser<Instruction> {
    #[inline]
    fn to_jmp(params: (Register, Expression)) -> Instruction {
        Instruction::Jmp {
            s: params.0,
            o: params.1,
        }
    }

    let arg = parser!(({offset_arg()}!![error!("expected offset")])->[to_jmp]);
    parser!({keyword(Keyword::Jmp)} .> arg)
}

fn link() -> impl Parser<Instruction> {
    parser!(({keyword(Keyword::Link)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_expr()})->[|(d, o)| Instruction::Link { d, o }])
}

macro_rules! ui_inst {
    ($name:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$inst)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_expr()})->[|(d, ui)| Instruction::$inst { d, ui }])
        }
    };
}

ui_inst!(ldui, LdUi);
ui_inst!(addpcui, AddPcUi);

macro_rules! br_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$kw)} .> {inst_req_expr()})->[|d| Instruction::$inst { d }])
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
br_inst!(jr, Jr, Jr);

macro_rules! mv_inst {
    ($name:ident, $kw:ident, $inst:ident) => {
        fn $name() -> impl Parser<Instruction> {
            parser!(({keyword(Keyword::$kw)} .> {alu_args()})->[|(d, l, r)| Instruction::$inst { d, l, r }])
        }
    };
}

mv_inst!(mvc, MvC, MvC);
mv_inst!(mvz, MvZ, MvZ);
mv_inst!(mvs, MvS, MvS);
mv_inst!(mvo, MvO, MvO);
mv_inst!(mvnc, MvNc, MvNc);
mv_inst!(mvnz, MvNz, MvNz);
mv_inst!(mvns, MvNs, MvNs);
mv_inst!(mvno, MvNo, MvNo);
mv_inst!(mveq, MvEq, MvZ);
mv_inst!(mvneq, MvNeq, MvNz);
mv_inst!(mvul, MvUL, MvNc);
mv_inst!(mvuge, MvUGe, MvC);
mv_inst!(mvule, MvULe, MvULe);
mv_inst!(mvug, MvUG, MvUG);
mv_inst!(mvsl, MvSL, MvSL);
mv_inst!(mvsge, MvSGe, MvSGe);
mv_inst!(mvsle, MvSLe, MvSLe);
mv_inst!(mvsg, MvSG, MvSG);

fn mov() -> impl Parser<Instruction> {
    parser!(({keyword(Keyword::Mov)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_reg()})->[|(d, s)| Instruction::Mov { d, s }])
}

fn ldi() -> impl Parser<Instruction> {
    parser!(({keyword(Keyword::LdI)} .> {inst_req_reg()} <. {inst_req_comma()} <.> {inst_req_expr()})->[|(d, s)| Instruction::LdI { d, s }])
}

fn inst() -> impl Parser<Instruction> {
    choice!(
        nop(),
        brk(),
        hlt(),
        err(),
        sys(),
        clrk(),
        add(),
        addc(),
        sub(),
        subb(),
        and(),
        or(),
        xor(),
        shl(),
        lsr(),
        asr(),
        mul(),
        cmp(),
        bit(),
        test(),
        inc(),
        incc(),
        dec(),
        decb(),
        neg(),
        negb(),
        not(),
        ld(),
        ld8(),
        ld8s(),
        ld16(),
        ld16s(),
        io_in(),
        st(),
        st8(),
        st16(),
        io_out(),
        jmp(),
        link(),
        ldui(),
        addpcui(),
        brc(),
        brz(),
        brs(),
        bro(),
        brnc(),
        brnz(),
        brns(),
        brno(),
        breq(),
        brneq(),
        brul(),
        bruge(),
        brule(),
        brug(),
        brsl(),
        brsge(),
        brsle(),
        brsg(),
        jr(),
        mvc(),
        mvz(),
        mvs(),
        mvo(),
        mvnc(),
        mvnz(),
        mvns(),
        mvno(),
        mveq(),
        mvneq(),
        mvul(),
        mvuge(),
        mvule(),
        mvug(),
        mvsl(),
        mvsge(),
        mvsle(),
        mvsg(),
        mov(),
        ldi(),
    )
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

pub fn parse_line(line: &[Token<TokenKind>], file_server: &FileServer) -> Result<Line, ParseError> {
    let input = TokenStream::new(line);

    let line_content = choice!(
        parser!({label_def()}->[LineKind::Label]),
        parser!({const_def()}->[|(name, expr)| LineKind::Define(name, expr)]),
        parser!({dir()}->[LineKind::Directive]),
        parser!({inst()}->[LineKind::Instruction])
    );

    let parser = parser!(line_content <. *{comment()} <. {eof()}!![
        error!(all "unexpected line continuation", all "this already forms a complete instruction")
    ]);

    match parser.run(input) {
        ParseResult::Match { value: kind, .. } => Ok(Line {
            kind,
            number: line
                .first()
                .unwrap()
                .span
                .start_pos()
                .line_column(file_server)
                .0 as usize,
            span: line.first().unwrap().span.join(&line.last().unwrap().span),
        }),
        ParseResult::NoMatch => {
            let full_span = line.first().unwrap().span.join(&line.last().unwrap().span);
            Err(ParseError::new(full_span, "invalid instruction"))
        }
        ParseResult::Err(err) => Err(err),
    }
}
