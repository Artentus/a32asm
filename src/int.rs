#[const_trait]
trait Integer {
    const BIT_COUNT: u32;
    const IS_SIGNED: bool;
}

macro_rules! impl_uint {
    ($it:ty : $bits:literal) => {
        impl const Integer for $it {
            const BIT_COUNT: u32 = $bits;
            const IS_SIGNED: bool = false;
        }
    };
}

impl_uint!(u8 : 8);
impl_uint!(u16 : 16);
impl_uint!(u32 : 32);
impl_uint!(u64 : 64);

macro_rules! impl_sint {
    ($it:ty : $bits:literal) => {
        impl const Integer for $it {
            const BIT_COUNT: u32 = $bits;
            const IS_SIGNED: bool = true;
        }
    };
}

impl_sint!(i8 : 8);
impl_sint!(i16 : 16);
impl_sint!(i32 : 32);
impl_sint!(i64 : 64);

struct If<const B: bool>;

trait True {}
impl True for If<true> {}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
struct SizedInt<I: Integer, const BIT_COUNT: u32>(I)
where
    If<{ BIT_COUNT <= I::BIT_COUNT }>: True,
    If<{ ((BIT_COUNT > 0) & !I::IS_SIGNED) | ((BIT_COUNT > 1) & I::IS_SIGNED) }>: True;

macro_rules! def_sized_int {
    ($name:ident : $it:ty[$bits:literal]) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(transparent)]
        pub struct $name(SizedInt<$it, $bits>);

        #[allow(dead_code)]
        impl $name {
            const I_MIN: $it = if $bits >= <$it as Integer>::BIT_COUNT {
                <$it>::MIN
            } else {
                if <$it as Integer>::IS_SIGNED {
                    (!2u128.pow($bits - 1) + 1) as $it
                } else {
                    0
                }
            };

            const I_MAX: $it = if $bits >= <$it as Integer>::BIT_COUNT {
                <$it>::MAX
            } else {
                if <$it as Integer>::IS_SIGNED {
                    (2u128.pow($bits - 1) - 1) as $it
                } else {
                    (2u128.pow($bits) - 1) as $it
                }
            };

            pub const MIN: Self = Self(SizedInt(Self::I_MIN));
            pub const MAX: Self = Self(SizedInt(Self::I_MAX));

            #[allow(unused_comparisons)]
            #[inline]
            pub const fn new(v: $it) -> Option<Self> {
                if (v >= Self::I_MIN) && (v <= Self::I_MAX) {
                    Some(Self(SizedInt(v)))
                } else {
                    None
                }
            }

            #[inline]
            pub const unsafe fn new_unchecked(v: $it) -> Self {
                Self(SizedInt(v))
            }

            #[allow(unused_comparisons)]
            #[inline]
            pub const fn new_clamp(v: $it) -> Self {
                if v < Self::I_MIN {
                    Self(SizedInt(Self::I_MIN))
                } else if v > Self::I_MAX {
                    Self(SizedInt(Self::I_MAX))
                } else {
                    Self(SizedInt(v))
                }
            }

            pub fn checked_add(self, rhs: Self) -> Option<Self> {
                let result = self.0 .0.checked_add(rhs.0 .0)?;
                Self::new(result)
            }

            pub fn checked_sub(self, rhs: Self) -> Option<Self> {
                let result = self.0 .0.checked_sub(rhs.0 .0)?;
                Self::new(result)
            }

            pub fn saturating_add(self, rhs: Self) -> Self {
                let result = self.0 .0.saturating_add(rhs.0 .0);
                Self::new_clamp(result)
            }

            pub fn saturating_sub(self, rhs: Self) -> Self {
                let result = self.0 .0.saturating_sub(rhs.0 .0);
                Self::new_clamp(result)
            }

            #[inline]
            pub const fn into_inner(self) -> $it {
                self.0 .0
            }
        }

        impl const Into<$it> for $name {
            #[inline]
            fn into(self) -> $it {
                self.0 .0
            }
        }

        impl std::fmt::Debug for $name {
            #[inline]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                std::fmt::Debug::fmt(&self.0 .0, f)
            }
        }

        impl std::fmt::Display for $name {
            #[inline]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                std::fmt::Display::fmt(&self.0 .0, f)
            }
        }

        #[allow(unused_macros)]
        macro_rules! $name {
            ($v:literal) => {
                #[allow(unused_comparisons)]
                {
                    static_assertions::const_assert!(
                        $v >= <$crate::int::$name as Into<$it>>::into($crate::int::$name::MIN)
                    );
                    static_assertions::const_assert!(
                        $v <= <$crate::int::$name as Into<$it>>::into($crate::int::$name::MAX)
                    );

                    unsafe { $crate::int::$name::new_unchecked($v) }
                }
            };
        }
    };
}

def_sized_int!(u5 : u8[5]);
def_sized_int!(i22 : i64[22]);
