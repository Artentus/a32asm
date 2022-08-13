use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

pub struct ValueSlice<T, const N: usize> {
    slice: [MaybeUninit<T>; N],
    len: usize,
}

impl<T, const N: usize> ValueSlice<T, N> {
    #[inline]
    const fn new(slice: [MaybeUninit<T>; N], len: usize) -> Self {
        assert!(len <= N);
        Self { slice, len }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }
}

impl<T, const N: usize> AsRef<[T]> for ValueSlice<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        unsafe { MaybeUninit::slice_assume_init_ref(&self.slice[..self.len]) }
    }
}

impl<T, const N: usize> AsMut<[T]> for ValueSlice<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.slice[..self.len]) }
    }
}

impl<T, const N: usize> Deref for ValueSlice<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { MaybeUninit::slice_assume_init_ref(&self.slice[..self.len]) }
    }
}

impl<T, const N: usize> DerefMut for ValueSlice<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.slice[..self.len]) }
    }
}

pub struct ValueSliceIter<T, const N: usize> {
    slice: [MaybeUninit<T>; N],
    len: usize,
    pos: usize,
}

impl<T, const N: usize> IntoIterator for ValueSlice<T, N> {
    type Item = T;
    type IntoIter = ValueSliceIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        ValueSliceIter {
            slice: self.slice,
            len: self.len,
            pos: 0,
        }
    }
}

impl<T, const N: usize> Iterator for ValueSliceIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.len {
            let item = unsafe { self.slice[self.pos].assume_init_read() };
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

pub struct ValueString<const N: usize> {
    string: [u8; N],
    len: usize,
}

impl<const N: usize> From<ValueSlice<char, N>> for ValueString<{ N * 4 }> {
    fn from(slice: ValueSlice<char, N>) -> Self {
        let mut string = [0; N * 4];
        let mut pos = 0;

        for c in slice {
            pos += c.encode_utf8(&mut string[pos..]).len();
        }

        ValueString { string, len: pos }
    }
}

impl<const N: usize> AsRef<str> for ValueString<N> {
    #[inline]
    fn as_ref(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.string[..self.len]) }
    }
}

impl<const N: usize> AsMut<str> for ValueString<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        unsafe { std::str::from_utf8_unchecked_mut(&mut self.string[..self.len]) }
    }
}

impl<const N: usize> Deref for ValueString<N> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::str::from_utf8_unchecked(&self.string[..self.len]) }
    }
}

impl<const N: usize> DerefMut for ValueString<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::str::from_utf8_unchecked_mut(&mut self.string[..self.len]) }
    }
}

pub trait IteratorEx: Iterator {
    fn next_n<const N: usize>(&mut self) -> ValueSlice<Self::Item, N> {
        let mut result = MaybeUninit::uninit_array();
        for i in 0..N {
            if let Some(item) = self.next() {
                result[i].write(item);
            } else {
                return ValueSlice::new(result, i);
            }
        }

        ValueSlice::new(result, N)
    }
}

impl<I: Iterator> IteratorEx for I {}
