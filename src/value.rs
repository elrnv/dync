//! This module defines a mechanism for type earasure simplar to `dyn Any`.
//!
//! This method allows storing and referencing values coming from a `VecDrop` inside other `std`
//! containers without needing a downcast.
//!
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::{ManuallyDrop, MaybeUninit};

// At the time of this writing, there is no evidence that there is a significant benefit in sharing
// vtables via Rc or Arc, but to make potential future refactoring easier we use the Ptr alias.
use std::boxed::Box as Ptr;

use crate::bytes::*;
pub use crate::copy_value::*;
use crate::traits::*;
use crate::vtable::*;

#[derive(Debug)]
pub enum Error {
    /// Value could not fit into a single pointer sized word.
    ValueTooLarge,
    /// Mismatched types.
    ///
    /// Trying to assign a value of one type to a value of another.
    MismatchedTypes { expected: TypeId, actual: TypeId },
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::ValueTooLarge => {
                write!(f, "Value could not fit into a single pointer sized word.\nTry constructing a BoxValue instead.")?;
            }
            Error::MismatchedTypes { expected, actual } => {
                writeln!(f, "Trying to assign a value of one type (with TypeId {:?}) to a value of another (with TypeId {:?}).", actual, expected)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

impl<T: Any> HasDrop for (DropFn, T) {
    #[inline]
    fn drop_fn(&self) -> &DropFn {
        &self.0
    }
}

// The following forwarding impls make it more straightforward to call the Has* methods on the
// (DropFn, V) vtable, although they are not strictly necessary.

impl<V: HasClone> HasClone for (DropFn, V) {
    #[inline]
    fn clone_fn(&self) -> &CloneFn {
        &self.1.clone_fn()
    }
    #[inline]
    fn clone_from_fn(&self) -> &CloneFromFn {
        &self.1.clone_from_fn()
    }
    #[inline]
    fn clone_into_raw_fn(&self) -> &CloneIntoRawFn {
        &self.1.clone_into_raw_fn()
    }
}

impl<V: HasHash> HasHash for (DropFn, V) {
    #[inline]
    fn hash_fn(&self) -> &HashFn {
        &self.1.hash_fn()
    }
}

impl<V: HasPartialEq> HasPartialEq for (DropFn, V) {
    #[inline]
    fn eq_fn(&self) -> &EqFn {
        &self.1.eq_fn()
    }
}

impl<V: HasEq> HasEq for (DropFn, V) {}

impl<V: HasDebug> HasDebug for (DropFn, V) {
    #[inline]
    fn fmt_fn(&self) -> &FmtFn {
        &self.1.fmt_fn()
    }
}

pub struct Value<B, V>
where
    B: GetBytesMut + DropAsAligned,
    V: ?Sized + HasDrop,
{
    pub(crate) bytes: ManuallyDrop<B>,
    pub(crate) type_id: TypeId,
    pub(crate) alignment: usize,
    pub(crate) vtable: ManuallyDrop<Ptr<V>>,
}

pub type SmallValue<V> = Value<MaybeUninit<usize>, V>;
pub type BoxValue<V> = Value<Box<[MaybeUninit<u8>]>, V>;

impl<V: HasDrop> SmallValue<V> {
    #[inline]
    pub fn try_new<T: Any + DropBytes>(value: T) -> Option<Value<MaybeUninit<usize>, V>>
    where
        V: VTable<T>,
    {
        // Prevent drop of value when it goes out of scope.
        let val = ManuallyDrop::new(value);
        val.try_into_usize().map(|usized_value| Value {
            bytes: ManuallyDrop::new(usized_value),
            type_id: TypeId::of::<T>(),
            alignment: std::mem::align_of::<T>(),
            vtable: ManuallyDrop::new(Ptr::new(V::build_vtable())),
        })
    }
    /// This function will panic if the given type does not fit into a `usize`.
    #[inline]
    pub fn new<T: Any + DropBytes>(value: T) -> Value<MaybeUninit<usize>, V>
    where
        V: VTable<T>,
    {
        Self::try_new(value).unwrap()
    }
}

impl<V: ?Sized + HasDrop> SmallValue<V> {
    /// Create a new `SmallValue` from a `usize` and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: MaybeUninit<usize>,
        type_id: TypeId,
        alignment: usize,
        vtable: Ptr<V>,
    ) -> Value<MaybeUninit<usize>, V> {
        Value {
            bytes: ManuallyDrop::new(bytes),
            type_id,
            alignment,
            vtable: ManuallyDrop::new(vtable),
        }
    }

    #[inline]
    pub fn upcast<U: HasDrop + From<V>>(self) -> SmallValue<U>
    where
        V: Clone,
    {
        // Inhibit drop for self, it will be dropped by the returned value
        let mut md = ManuallyDrop::new(self);
        let output = Value {
            bytes: md.bytes,
            type_id: md.type_id,
            alignment: md.alignment,
            vtable: ManuallyDrop::new(Ptr::new(U::from((**md.vtable).clone()))),
        };

        // Manually drop the old vtable.
        // This is safe since the old vtable will not be used again since self is consumed.
        // It will also not be double dropped since it is cloned above.
        unsafe {
            ManuallyDrop::drop(&mut md.vtable);
        }

        output
    }

    /// Convert this value into its destructured parts.
    ///
    /// The caller must insure that the memory allocated by the returned bytes is freed.
    #[inline]
    pub fn into_raw_parts(self) -> (MaybeUninit<usize>, TypeId, usize, Ptr<V>) {
        // Inhibit drop for self.
        let mut md = ManuallyDrop::new(self);

        // Pass ownership of the vtable and bytes to the caller. This allows the table to be
        // dropped automatically if the caller ignores it, however the data represented by bytes
        // is leaked.
        let vtable = unsafe { ManuallyDrop::take(&mut md.vtable) };
        let bytes = unsafe { ManuallyDrop::take(&mut md.bytes) };

        (bytes, md.type_id, md.alignment, vtable)
    }
}

impl<V: HasDrop> BoxValue<V> {
    #[inline]
    pub fn new<T: Any + DropBytes>(value: T) -> Value<Box<[MaybeUninit<u8>]>, V>
    where
        V: VTable<T>,
    {
        Value {
            bytes: ManuallyDrop::new(Bytes::box_into_box_bytes(Box::new(value))),
            type_id: TypeId::of::<T>(),
            alignment: std::mem::align_of::<T>(),
            vtable: ManuallyDrop::new(Ptr::new(V::build_vtable())),
        }
    }
}

impl<V: ?Sized + HasDrop> BoxValue<V> {
    /// Create a new `BoxValue` from boxed bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: Box<[MaybeUninit<u8>]>,
        type_id: TypeId,
        alignment: usize,
        vtable: Ptr<V>,
    ) -> Value<Box<[MaybeUninit<u8>]>, V> {
        Value {
            bytes: ManuallyDrop::new(bytes),
            type_id,
            alignment,
            vtable: ManuallyDrop::new(vtable),
        }
    }

    #[inline]
    pub fn upcast<U: HasDrop + From<V>>(self) -> BoxValue<U>
    where
        V: Clone,
    {
        // Inhibit drop for self, it will be dropped by the returned value
        let mut md = ManuallyDrop::new(self);
        // It is safe to take the bytes from md since self is consumed and will not be used later.
        let output = Value {
            bytes: ManuallyDrop::new(unsafe { ManuallyDrop::take(&mut md.bytes) }),
            type_id: md.type_id,
            alignment: md.alignment,
            vtable: ManuallyDrop::new(Ptr::new(U::from((**md.vtable).clone()))),
        };

        // Manually drop the old vtable.
        // This is safe since the old vtable will not be used again since self is consumed.
        // It will also not be double dropped since it is cloned above.
        unsafe {
            ManuallyDrop::drop(&mut md.vtable);
        }

        output
    }

    /// Convert this value into its destructured parts.
    ///
    /// The caller must insure that the memory allocated by the returned bytes is freed.
    #[inline]
    pub fn into_raw_parts(self) -> (Box<[MaybeUninit<u8>]>, TypeId, usize, Ptr<V>) {
        // Inhibit drop for self.
        let mut md = ManuallyDrop::new(self);

        // Pass ownership of the vtable and bytes to the caller. This allows the table to be
        // dropped automatically if the caller ignores it, however the data represented by bytes
        // is leaked.
        let vtable = unsafe { ManuallyDrop::take(&mut md.vtable) };
        let bytes = unsafe { ManuallyDrop::take(&mut md.bytes) };

        (bytes, md.type_id, md.alignment, vtable)
    }
}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop> Value<B, V> {
    #[inline]
    pub fn as_ref(&self) -> ValueRef<V> {
        ValueRef {
            bytes: self.bytes.get_bytes_ref(),
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Ref(&self.vtable),
        }
    }
    #[inline]
    pub fn as_mut(&mut self) -> ValueMut<V> {
        ValueMut {
            bytes: self.bytes.get_bytes_mut(),
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Ref(&self.vtable),
        }
    }
}

unsafe impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop + HasSend> Send for Value<B, V> {}
unsafe impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop + HasSync> Sync for Value<B, V> {}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDebug + HasDrop> fmt::Debug for Value<B, V> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe { self.vtable.fmt_fn()(self.bytes.get_bytes_ref(), f) }
    }
}

impl<V: ?Sized + Clone + HasClone + HasDrop> Clone for Value<MaybeUninit<usize>, V> {
    #[inline]
    fn clone(&self) -> Value<MaybeUninit<usize>, V> {
        self.as_ref().clone_small_value()
    }
}

impl<V: ?Sized + Clone + HasClone + HasDrop> Clone for Value<Box<[MaybeUninit<u8>]>, V> {
    #[inline]
    fn clone(&self) -> Value<Box<[MaybeUninit<u8>]>, V> {
        self.as_ref().clone_value()
    }
}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop> Drop for Value<B, V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // This is safe since self will never be used after this call.
            self.vtable.drop_fn()(self.bytes.get_bytes_mut());

            // Manually drop what we promised.
            self.bytes.drop_as_aligned(self.alignment);
            ManuallyDrop::drop(&mut self.vtable);
        }
    }
}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop + HasPartialEq> PartialEq for Value<B, V> {
    /// # Panics
    ///
    /// This function panics if the types of the two operands don't match.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(
            self.type_id, other.type_id,
            "Comparing values of different types is forbidden"
        );
        // This is safe because we have just checked that the two types are the same.
        unsafe { self.vtable.eq_fn()(self.bytes.get_bytes_ref(), other.bytes.get_bytes_ref()) }
    }
}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop + HasPartialEq> Eq for Value<B, V> {}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop + HasHash> Hash for Value<B, V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            self.vtable.hash_fn()(self.bytes.get_bytes_ref(), state);
        }
    }
}

impl<B: GetBytesMut + DropAsAligned, V: ?Sized + HasDrop> Value<B, V> {
    impl_value_base!();
}

impl<V: ?Sized + HasDrop> Value<Box<[MaybeUninit<u8>]>, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<Box<T>> {
        // Override the Drop implementation, since we don't actually want to drop the contents
        // here, just the vtable.
        let mut s = ManuallyDrop::new(self);
        let output = if s.is::<T>() {
            // This is safe since we check that self.bytes represents a `T`.
            Some(unsafe { Bytes::box_from_box_bytes(ManuallyDrop::take(&mut s.bytes)) })
        } else {
            None
        };
        // We are done with self, dropping the remaining fields here to avoid memory leaks.
        // This is safe since `s` will not be used again.
        unsafe {
            // Note that TypeId is a Copy type and does not need to be manually dropped.
            ManuallyDrop::drop(&mut s.vtable);
        }
        output
    }
}

impl<V: ?Sized + HasDrop> Value<MaybeUninit<usize>, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<T> {
        // Override the Drop implementation, since we don't actually want to drop the contents
        // here, just the vtable.
        let mut s = ManuallyDrop::new(self);
        // This is safe since we check that self.bytes represent a `T`.
        let output = if s.is::<T>() {
            unsafe { Bytes::try_from_usize(ManuallyDrop::take(&mut s.bytes)) }
        } else {
            None
        };
        // We are done with self, dropping the remaining fields here to avoid memory leaks.
        // This is safe since `s` will not be used again.
        unsafe {
            // Note that TypeId is a Copy type and does not need to be manually dropped.
            ManuallyDrop::drop(&mut s.vtable);
        }
        output
    }
}

macro_rules! impl_value_ref_traits {
    ($value_ref:ident : $($maybe_drop:ident)*) => {
        unsafe impl<'a, V: ?Sized + HasSend $( + $maybe_drop)*> Send for $value_ref<'a, V> {}
        unsafe impl<'a, V: ?Sized + HasSync $( + $maybe_drop)*> Sync for $value_ref<'a, V> {}

        impl<'a, V: ?Sized + HasHash $( + $maybe_drop)*> Hash for $value_ref<'a, V> {
            #[inline]
            fn hash<H: Hasher>(&self, state: &mut H) {
                unsafe {
                    self.vtable.as_ref().hash_fn()(self.bytes.get_bytes_ref(), state);
                }
            }
        }

        impl<'a, V: ?Sized + HasDebug $( + $maybe_drop)*> fmt::Debug for $value_ref<'a, V> {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                unsafe { self.vtable.as_ref().fmt_fn()(self.bytes.get_bytes_ref(), f) }
            }
        }

        impl<'a, V: ?Sized + HasPartialEq $( + $maybe_drop)*> PartialEq for $value_ref<'a, V> {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                assert_eq!(
                    self.type_id, other.type_id,
                    "Comparing values of different types is forbidden"
                );
                // This is safe because we have just checked that the two types are the same.
                unsafe {
                    self.vtable.as_ref().eq_fn()(self.bytes.get_bytes_ref(), other.bytes.get_bytes_ref())
                }
            }
        }

        impl<'a, V: ?Sized + HasEq $( + $maybe_drop)*> Eq for $value_ref<'a, V> {}
    }
}

// Impl value ref traits for CopyValue types.
impl_value_ref_traits!(CopyValueMut:);
impl_value_ref_traits!(CopyValueRef:);

/// A generic value reference into a buffer.
#[derive(Clone)]
pub struct ValueRef<'a, V>
where
    V: ?Sized + HasDrop,
{
    pub(crate) bytes: &'a [MaybeUninit<u8>],
    pub(crate) type_id: TypeId,
    pub(crate) alignment: usize,
    pub(crate) vtable: VTableRef<'a, V>,
}

impl_value_ref_traits!(ValueRef: HasDrop);

impl<'a, V: HasDrop> ValueRef<'a, V> {
    /// Create a new `ValueRef` from a typed reference.
    #[inline]
    pub fn new<T: Any + DropBytes>(typed: &'a T) -> ValueRef<'a, V>
    where
        V: VTable<T>,
    {
        // Reminder: We need DropFn here in case the referenced value is cloned.
        ValueRef {
            bytes: typed.as_bytes(),
            type_id: TypeId::of::<T>(),
            alignment: std::mem::align_of::<T>(),
            vtable: VTableRef::Box(Box::new(V::build_vtable())),
        }
    }
}

impl<'a, B: GetBytesMut + DropAsAligned, V> From<&'a Value<B, V>> for ValueRef<'a, V>
where
    B: GetBytesRef,
    V: ?Sized + Clone + HasDrop,
{
    fn from(val: &'a Value<B, V>) -> Self {
        ValueRef {
            bytes: val.bytes.get_bytes_ref(),
            type_id: val.type_id,
            alignment: val.alignment,
            vtable: Ptr::clone(&val.vtable).into(),
        }
    }
}

impl<'a, V: ?Sized + HasDrop> ValueRef<'a, V> {
    impl_value_base!();

    /// Create a new `ValueRef` from a slice of bytes and an associated `TypeId`.
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
    ) -> ValueRef<'a, V> {
        ValueRef {
            bytes,
            type_id,
            alignment,
            vtable: vtable.into(),
        }
    }

    /// Clone the referenced value.
    ///
    /// Unlike the `Clone` trait, this function will produce an owned clone of the value pointed to
    /// by this value reference.
    #[inline]
    pub fn clone_value(&self) -> Value<Box<[MaybeUninit<u8>]>, V>
    where
        V: HasClone + Clone,
    {
        Value {
            bytes: ManuallyDrop::new(unsafe { self.vtable.as_ref().clone_fn()(&self.bytes) }),
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: ManuallyDrop::new(Ptr::from(self.vtable.as_ref().clone())),
        }
    }

    /// Clone the referenced value.
    ///
    /// Unlike the `Clone` trait, this function will produce an owned clone of the value pointed to
    /// by this value reference.
    ///
    /// This version of `clone_value` tries to fit the underlying value into a `usize` sized value,
    /// and panics if that fails.
    #[inline]
    pub fn clone_small_value(&self) -> Value<MaybeUninit<usize>, V>
    where
        V: HasClone + Clone,
    {
        let mut bytes = MaybeUninit::uninit();
        // This is safe because clone_into_raw_fn will not attempt to drop the value at the
        // destination.
        unsafe {
            self.vtable.clone_into_raw_fn()(self.bytes.get_bytes_ref(), bytes.as_bytes_mut());
        }
        Value {
            bytes: ManuallyDrop::new(bytes),
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: ManuallyDrop::new(Ptr::from(self.vtable.as_ref().clone())),
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }

    #[inline]
    pub fn upcast<U: ?Sized + HasDrop + From<V>>(self) -> ValueRef<'a, U>
    where
        V: Clone,
    {
        ValueRef {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from(self.vtable.take()))),
        }
    }

    #[inline]
    pub fn upcast_ref<U: ?Sized + HasDrop + From<V>>(&self) -> ValueRef<U>
    where
        V: Clone,
    {
        let vtable = self.vtable.as_ref();
        ValueRef {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from((*vtable).clone()))),
        }
    }

    /// An alternative to `Clone` that doesn't require cloning the vtable.
    #[inline]
    pub fn reborrow(&self) -> ValueRef<V> {
        ValueRef {
            bytes: &*self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }
}

/// A generic mutable value reference into a buffer.
pub struct ValueMut<'a, V>
where
    V: ?Sized + HasDrop,
{
    pub(crate) bytes: &'a mut [MaybeUninit<u8>],
    pub(crate) type_id: TypeId,
    pub(crate) alignment: usize,
    pub(crate) vtable: VTableRef<'a, V>,
}

impl_value_ref_traits!(ValueMut: HasDrop);

impl<'a, V: HasDrop> ValueMut<'a, V> {
    /// Create a new `ValueRef` from a typed reference.
    #[inline]
    pub fn new<T: Any>(typed: &'a mut T) -> ValueMut<'a, V>
    where
        V: VTable<T>,
    {
        ValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
            alignment: std::mem::align_of::<T>(),
            vtable: VTableRef::Box(Box::new(V::build_vtable())),
        }
    }
}

impl<'a, V: ?Sized + HasDrop> ValueMut<'a, V> {
    impl_value_base!();

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap<'b>(&mut self, other: &mut ValueMut<'b, V>) {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.swap_with_slice(other.bytes);
        }
    }

    /// Moves the given value into self.
    pub fn assign<B: GetBytesMut + DropAsAligned>(&mut self, mut value: Value<B, V>) {
        // Swapping the values ensures that `value` will not be dropped, but the overwritten value
        // in self will.
        self.swap(&mut value.as_mut())
    }

    /// Clone `other` into `self`.
    ///
    /// This function will call `drop` on any values stored in `self`.
    #[inline]
    pub fn clone_from_other<'b>(&mut self, other: impl Into<ValueRef<'b, V>>) -> Result<(), Error>
    where
        V: HasClone + 'b,
    {
        let other = other.into();
        if self.value_type_id() == other.value_type_id() {
            unsafe {
                // We are cloning other.bytes into self.bytes.
                // This function will call the appropriate typed clone_from function, which will
                // automatically drop the previous value of self.bytes, so no manual drop call is
                // needed here.
                self.vtable.as_ref().clone_from_fn()(&mut self.bytes, other.bytes);
            }
            Ok(())
        } else {
            Err(Error::MismatchedTypes {
                expected: self.value_type_id(),
                actual: other.value_type_id(),
            })
        }
    }

    /// Clone the referenced value.
    ///
    /// Unlike the `Clone` trait, this function will produce an owned clone of the value pointed to
    /// by this value reference.
    #[inline]
    pub fn clone_value(&self) -> Value<Box<[MaybeUninit<u8>]>, V>
    where
        V: HasClone + Clone,
    {
        Value {
            bytes: ManuallyDrop::new(unsafe { self.vtable.as_ref().clone_fn()(&self.bytes) }),
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: ManuallyDrop::new(Ptr::from(self.vtable.as_ref().clone())),
        }
    }

    /// Clone the referenced value.
    ///
    /// Unlike the `Clone` trait, this function will produce an owned clone of the value pointed to
    /// by this value reference.
    ///
    /// This version of `clone_value` tries to fit the underlying value into a `usize` sized value,
    /// and panics if that fails.
    #[inline]
    pub fn clone_small_value(&self) -> Value<MaybeUninit<usize>, V>
    where
        V: HasClone + Clone,
    {
        let mut bytes = MaybeUninit::uninit();
        // This is safe because clone_into_raw_fn will not attempt to drop the value at the
        // destination.
        unsafe {
            self.vtable.clone_into_raw_fn()(self.bytes.get_bytes_ref(), bytes.as_bytes_mut());
        }
        Value {
            bytes: ManuallyDrop::new(bytes),
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: ManuallyDrop::new(Ptr::from(self.vtable.as_ref().clone())),
        }
    }

    /// Create a new `ValueMut` from a slice of bytes and an associated `TypeId`.
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
    ) -> ValueMut<'a, V> {
        ValueMut {
            bytes,
            type_id,
            alignment,
            vtable: vtable.into(),
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
    }

    /// A consuming upcast that carries the lifetime of the underlying value reference.
    #[inline]
    pub fn upcast<U: ?Sized + HasDrop + From<V>>(self) -> ValueMut<'a, U>
    where
        V: Clone,
    {
        ValueMut {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from(self.vtable.take()))),
        }
    }

    /// Create a new mutable value reference upcast from the current one.
    #[inline]
    pub fn upcast_mut<U: ?Sized + HasDrop + From<V>>(&mut self) -> ValueMut<U>
    where
        V: Clone,
    {
        ValueMut {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Box(Box::new(U::from((*self.vtable).clone()))),
        }
    }

    #[inline]
    pub fn reborrow(&self) -> ValueRef<V> {
        ValueRef {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    #[inline]
    pub fn reborrow_mut(&mut self) -> ValueMut<V> {
        ValueMut {
            bytes: self.bytes,
            type_id: self.type_id,
            alignment: self.alignment,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }
}

/*
 * Valid conversions.
 */

impl<'a, V: HasDrop> From<ValueMut<'a, V>> for ValueRef<'a, V> {
    #[inline]
    fn from(v: ValueMut<'a, V>) -> ValueRef<'a, V> {
        ValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
            alignment: v.alignment,
            vtable: v.vtable,
        }
    }
}

/// Implementation for dropping a copy.
///
/// This enables the following two conversions.
unsafe fn drop_copy(_: &mut [MaybeUninit<u8>]) {}

impl<'a, V: Any + Clone> From<CopyValueMut<'a, V>> for ValueMut<'a, (DropFn, V)> {
    #[inline]
    fn from(v: CopyValueMut<'a, V>) -> ValueMut<'a, (DropFn, V)> {
        ValueMut {
            bytes: v.bytes,
            type_id: v.type_id,
            alignment: v.alignment,
            vtable: VTableRef::Box(Box::new((drop_copy, v.vtable.take()))),
        }
    }
}

impl<'a, V: Any + Clone> From<CopyValueRef<'a, V>> for ValueRef<'a, (DropFn, V)> {
    #[inline]
    fn from(v: CopyValueRef<'a, V>) -> ValueRef<'a, (DropFn, V)> {
        ValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
            alignment: v.alignment,
            vtable: VTableRef::Box(Box::new((drop_copy, v.vtable.take()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dync_trait;
    use std::rc::Rc;

    #[dync_trait(dync_crate_name = "crate")]
    pub trait Val: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + 'static {}
    impl<T> Val for T where T: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + 'static {}

    #[test]
    #[should_panic]
    fn forbidden_value_compare() {
        let a = BoxValue::<ValVTable>::new(Rc::new("Hello"));
        let b = BoxValue::<ValVTable>::new(Rc::new(String::from("Hello")));
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic]
    fn forbidden_value_ref_compare() {
        let a = BoxValue::<ValVTable>::new(Rc::new("Hello"));
        let b = BoxValue::<ValVTable>::new(Rc::new(String::from("Hello")));
        assert_eq!(a.as_ref(), b.as_ref());
    }

    #[test]
    #[should_panic]
    fn forbidden_value_mut_compare() {
        let mut a = BoxValue::<ValVTable>::new(Rc::new("Hello"));
        let mut b = BoxValue::<ValVTable>::new(Rc::new(String::from("Hello")));
        assert_eq!(a.as_mut(), b.as_mut());
    }

    #[test]
    fn value_equality() {
        let a = Rc::new(String::from("Hello"));
        let b = Rc::new(String::from("Hello"));
        assert_eq!(&a, &b);

        let a = BoxValue::<ValVTable>::new(Rc::new(String::from("Hello")));
        let b = BoxValue::<ValVTable>::new(Rc::new(String::from("Hello")));
        let c = b.clone();
        let c_rc = b.clone().downcast::<Rc<String>>().unwrap();
        let d = BoxValue::<ValVTable>::new(Rc::clone(&*c_rc));
        assert_eq!(&a, &b);
        assert_eq!(&a, &c);
        assert_eq!(&a, &d);
    }

    // This test checks that cloning and dropping clones works correctly.
    #[test]
    fn clone_test() {
        let val = BoxValue::<ValVTable>::new(Rc::new(1u8));
        assert_eq!(&val, &val.clone());
    }

    // This test checks that cloning and dropping clones works correctly.
    #[test]
    fn clone_small_test() {
        let val = SmallValue::<ValVTable>::new(Rc::new(1u8));
        assert_eq!(&val, &val.clone());
    }

    // Test that we can convert from a copy value to a regular value.
    #[test]
    fn copy_value_to_value_convert() {
        // make a vector and get a reference value from it.
        let v = crate::vec_copy::VecCopy::from(vec![1u32, 2, 3]);
        let copy_val = v.get_ref(1);
        let val: ValueRef<(DropFn, ())> = copy_val.into();
        assert_eq!(val.downcast::<u32>().unwrap(), &2u32);
    }
}
