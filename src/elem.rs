//!
//! This type defines types and traits useful for representing elements in untyped containers.
//!

use std::{
    any::{Any, TypeId},
    mem::{align_of, size_of},
};

macro_rules! generate_aligned_types {
    ($($t:ident($n:expr),)*) => {
	$(
	    #[derive(Copy, Clone, Debug, Default)]
	    #[repr(align($n))]
	    pub(crate) struct $t(u8);
	)*
    }
}

generate_aligned_types!(
    T0(1),
    T1(2),
    T2(4),
    T3(8),
    T4(16),
    T5(32),
    T6(64),
    T7(128),
    T8(256),
    T9(512),
    T10(1024),
    T11(2048),
    T12(4096),
    T13(8192),
    T14(16384),
    T15(32768),
    T16(65536),
    T17(131072),
    T18(262144),
    T19(524288),
    T20(1048576),
    T21(2097152),
    T22(4194304),
    T23(8388608),
    T24(16777216),
    T25(33554432),
    T26(67108864),
    T27(134217728),
    T28(268435456),
    T29(536870912),
);

#[macro_export]
macro_rules! eval_align {
    ($a:expr; $fn:ident ::<_ $(,$params:ident)*>($($args:expr),*) ) => {
	match $a {
	    1         => $fn::<T0, $($params,)*>($($args,)*),
	    2         => $fn::<T1, $($params,)*>($($args,)*),
	    4         => $fn::<T2, $($params,)*>($($args,)*),
	    8         => $fn::<T3, $($params,)*>($($args,)*),
	    16        => $fn::<T4, $($params,)*>($($args,)*),
	    32        => $fn::<T5, $($params,)*>($($args,)*),
	    64        => $fn::<T6, $($params,)*>($($args,)*),
	    128       => $fn::<T7, $($params,)*>($($args,)*),
	    256       => $fn::<T8, $($params,)*>($($args,)*),
	    512       => $fn::<T9, $($params,)*>($($args,)*),
	    1024      => $fn::<T10, $($params,)*>($($args,)*),
	    2048      => $fn::<T11, $($params,)*>($($args,)*),
	    4096      => $fn::<T12, $($params,)*>($($args,)*),
	    8192      => $fn::<T13, $($params,)*>($($args,)*),
	    16384     => $fn::<T14, $($params,)*>($($args,)*),
	    32768     => $fn::<T15, $($params,)*>($($args,)*),
	    65536     => $fn::<T16, $($params,)*>($($args,)*),
	    131072    => $fn::<T17, $($params,)*>($($args,)*),
	    262144    => $fn::<T18, $($params,)*>($($args,)*),
	    524288    => $fn::<T19, $($params,)*>($($args,)*),
	    1048576   => $fn::<T20, $($params,)*>($($args,)*),
	    2097152   => $fn::<T21, $($params,)*>($($args,)*),
	    4194304   => $fn::<T22, $($params,)*>($($args,)*),
	    8388608   => $fn::<T23, $($params,)*>($($args,)*),
	    16777216  => $fn::<T24, $($params,)*>($($args,)*),
	    33554432  => $fn::<T25, $($params,)*>($($args,)*),
	    67108864  => $fn::<T26, $($params,)*>($($args,)*),
	    134217728 => $fn::<T27, $($params,)*>($($args,)*),
	    268435456 => $fn::<T28, $($params,)*>($($args,)*),
	    536870912 => $fn::<T29, $($params,)*>($($args,)*),
	    _ => unreachable!("Unsupported alignment detected")
	}
    }
}

pub trait CopyElem: Any + Copy {}
impl<T> CopyElem for T where T: Any + Copy {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ElemInfo {
    /// Type encoding for hiding the type of data from the compiler.
    pub(crate) type_id: TypeId,
    /// Number of alignment chunks occupied by an element of this buffer.
    ///
    /// The size of the element in bytes is then given by `size * alignment`.
    pub(crate) size: usize,
    /// Alignment info for an element stored in this `Vec`.
    pub(crate) alignment: usize,
}

impl ElemInfo {
    #[inline]
    pub fn new<T: 'static>() -> ElemInfo {
        ElemInfo {
            type_id: TypeId::of::<T>(),
            size: size_of::<T>() / align_of::<T>(),
            alignment: align_of::<T>(),
        }
    }

    #[inline]
    pub const fn num_bytes(self) -> usize {
        self.size * self.alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_struct() {
        let elem_u32 = ElemInfo::new::<u32>();
        assert_eq!(
            elem_u32,
            ElemInfo {
                type_id: TypeId::of::<u32>(),
                size: 1,
                alignment: 4,
            }
        );

        assert_eq!(elem_u32.num_bytes(), 4);
    }
}
