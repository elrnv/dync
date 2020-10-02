use std::any::TypeId;
use std::mem::MaybeUninit;

use crate::bytes::*;
use crate::vtable::*;
use crate::CopyElem;

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

// Implement the basis for all value types.
macro_rules! impl_value_base {
    () => {
        /// Get the size of the value pointed-to by this reference.
        #[inline]
        pub fn size(&self) -> usize {
            self.bytes.get_bytes_ref().len()
        }

        /// Get the `TypeId` of the referenced value.
        #[inline]
        pub fn value_type_id(&self) -> TypeId {
            self.type_id
        }

        /// Get the alignment of the referenced value.
        #[inline]
        pub fn value_alignment(&self) -> usize {
            self.alignment
        }

        /// Returns `true` if this referenced value's type is the same as `T`.
        #[inline]
        pub fn is<T: 'static>(&self) -> bool {
            self.value_type_id() == TypeId::of::<T>()
        }

        // Check that this value represents the given type, and if so return the bytes.
        // This is a helper for downcasts.
        #[inline]
        fn downcast_with<T: 'static, U, F: FnOnce(Self) -> U>(self, f: F) -> Option<U> {
            if self.is::<T>() {
                Some(f(self))
            } else {
                None
            }
        }
    };
}

// Helper trait for calling trait functions that take bytes.
pub trait GetBytesRef {
    fn get_bytes_ref(&self) -> &[MaybeUninit<u8>];
}
pub trait GetBytesMut: GetBytesRef {
    fn get_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>];
}

// Bytes cannot be dropped willy-nilly. Dropping triggers deallocators, and they must know about
// about the size and alignment of the original allocation.
pub trait DropAsAligned {
    fn drop_as_aligned(&mut self, alignment: usize);
}

impl GetBytesRef for Box<[MaybeUninit<u8>]> {
    #[inline]
    fn get_bytes_ref(&self) -> &[MaybeUninit<u8>] {
        &*self
    }
}
impl GetBytesMut for Box<[MaybeUninit<u8>]> {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut *self
    }
}

impl DropAsAligned for Box<[MaybeUninit<u8>]> {
    #[inline]
    fn drop_as_aligned(&mut self, alignment: usize) {
        fn drop_bytes<T: 'static>(b: Box<[MaybeUninit<u8>]>) {
            let _ = unsafe { Box::from_raw(Box::into_raw(b) as *mut T) };
        }
        eval_align!(alignment; drop_bytes::<_>(std::mem::take(self)));
    }
}

impl GetBytesRef for MaybeUninit<usize> {
    #[inline]
    fn get_bytes_ref(&self) -> &[MaybeUninit<u8>] {
        self.as_bytes()
    }
}
impl GetBytesMut for MaybeUninit<usize> {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.as_bytes_mut()
    }
}

impl DropAsAligned for MaybeUninit<usize> {
    #[inline]
    fn drop_as_aligned(&mut self, _: usize) {
        // Value lives on the stack, no fancy deallocation necessary.
    }
}

impl GetBytesRef for [MaybeUninit<u8>] {
    #[inline]
    fn get_bytes_ref(&self) -> &[MaybeUninit<u8>] {
        self
    }
}
impl GetBytesMut for [MaybeUninit<u8>] {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self
    }
}

/// A generic value reference to a `Copy` type.
#[derive(Clone)]
pub struct CopyValueRef<'a, V = ()>
where
    V: ?Sized,
{
    pub(crate) bytes: &'a [MaybeUninit<u8>],
    pub(crate) type_id: TypeId,
    pub(crate) alignment: usize,
    pub(crate) vtable: VTableRef<'a, V>,
}

impl<'a, V> CopyValueRef<'a, V> {
    /// Create a new `CopyValueRef` from a typed reference.
    #[inline]
    pub fn new<T: CopyElem>(typed: &'a T) -> CopyValueRef<'a, V>
    where
        V: VTable<T>,
    {
        CopyValueRef {
            bytes: typed.as_bytes(),
            type_id: TypeId::of::<T>(),
            alignment: std::mem::align_of::<T>(),
            vtable: VTableRef::Box(Box::new(V::build_vtable())),
        }
    }
}

impl<'a, V: ?Sized> CopyValueRef<'a, V> {
    impl_value_base!();

    /// Create a new `CopyValueref` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: &'a [MaybeUninit<u8>],
        type_id: TypeId,
        alignment: usize,
        vtable: impl Into<VTableRef<'a, V>>,
    ) -> CopyValueRef<'a, V> {
        CopyValueRef {
            bytes,
            type_id,
            alignment,
            vtable: vtable.into(),
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: CopyElem>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }

    #[inline]
    pub fn upcast<U: ?Sized + From<V>>(self) -> CopyValueRef<'a, U>
    where
        V: Clone,
    {
        CopyValueRef {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from(self.vtable.take()))),
        }
    }

    #[inline]
    pub fn upcast_ref<U: ?Sized + From<V>>(&self) -> CopyValueRef<U>
    where
        V: Clone,
    {
        let vtable = self.vtable.as_ref();
        CopyValueRef {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from((*vtable).clone()))),
        }
    }
}

/// A generic mutable `Copy` value reference.
pub struct CopyValueMut<'a, V = ()>
where
    V: ?Sized,
{
    pub(crate) bytes: &'a mut [MaybeUninit<u8>],
    pub(crate) type_id: TypeId,
    pub(crate) alignment: usize,
    pub(crate) vtable: VTableRef<'a, V>,
}

impl<'a, V> CopyValueMut<'a, V> {
    /// Create a new `CopyValueMut` from a typed mutable reference.
    #[inline]
    pub fn new<T: CopyElem>(typed: &'a mut T) -> CopyValueMut<'a, V>
    where
        V: VTable<T>,
    {
        CopyValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
            alignment: std::mem::align_of::<T>(),
            vtable: VTableRef::Box(Box::new(V::build_vtable())),
        }
    }
}

impl<'a, V: ?Sized> CopyValueMut<'a, V> {
    impl_value_base!();

    /// Create a new `CopyValueMut` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: &'a mut [MaybeUninit<u8>],
        type_id: TypeId,
        alignment: usize,
        vtable: impl Into<VTableRef<'a, V>>,
    ) -> CopyValueMut<'a, V> {
        CopyValueMut {
            bytes,
            type_id,
            alignment,
            vtable: vtable.into(),
        }
    }

    /// Copy value from `other` to `self` and return `Self`.
    ///
    /// This function returns `None` if the values have different types.
    #[inline]
    pub fn copy(self, other: CopyValueRef<'a, V>) -> Option<Self> {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.copy_from_slice(other.bytes);
            Some(self)
        } else {
            None
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap(&mut self, other: &mut CopyValueMut<V>) {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.swap_with_slice(&mut other.bytes);
            assert_eq!(self.alignment, other.alignment);
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: CopyElem>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
    }

    #[inline]
    pub fn upcast<U: From<V>>(&mut self) -> CopyValueMut<U>
    where
        V: Clone,
    {
        let vtable = self.vtable.as_ref();
        CopyValueMut {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from(vtable.clone()))),
        }
    }
}

impl<'a, V> From<CopyValueMut<'a, V>> for CopyValueRef<'a, V> {
    #[inline]
    fn from(v: CopyValueMut<'a, V>) -> CopyValueRef<'a, V> {
        CopyValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
            alignment: v.alignment,
            vtable: v.vtable,
        }
    }
}
