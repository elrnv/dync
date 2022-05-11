use std::any::TypeId;
use std::mem::MaybeUninit;

use crate::bytes::*;
use crate::elem::*;
use crate::vtable::*;
use crate::CopyElem;

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

// Bytes cannot be dropped willy-nilly. Dropping triggers deallocators, and they must know
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
            // Box value is created from a Box<T> -> Box<[MaybeUninit<u8>].
            // Here T has the right alignment but often wrong size (e.g. if original T was an array).
            // Box<[MaybeUninit<u8>] has the right size but wrong alignment.
            // To ensure that the right number of elements is dropped with the right alignment
            // We need to cast to T and build a slice of the right size. Then drop that box.
            let size = b.len() / std::mem::align_of::<T>();
            let ptr: *mut T = Box::into_raw(b) as *mut T;
            unsafe {
                let slice_t = std::slice::from_raw_parts_mut(ptr, size);
                let _ = Box::from_raw(slice_t);
            }
        }
        eval_align!(alignment; drop_bytes::<_>(std::mem::take(self)));
    }
}

impl GetBytesRef for MaybeUninit<usize> {
    #[inline]
    fn get_bytes_ref(&self) -> &[MaybeUninit<u8>] {
        Bytes::as_bytes(self)
    }
}
impl GetBytesMut for MaybeUninit<usize> {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        Bytes::as_bytes_mut(self)
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
            self.bytes.swap_with_slice(other.bytes);
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
