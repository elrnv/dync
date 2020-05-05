//! This module defines a mechanism for type earasure simplar to `dyn Any`.
//!
//! This method allows storing and referencing values coming from a `DataBuffer` in other `std`
//! containers without needing a downcast.
//!
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use crate::bytes::*;
use crate::traits::*;
use crate::Elem;

#[derive(Debug)]
pub enum Error {
    /// Value could not fit into a single pointer sized word.
    ValueTooLarge,
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::ValueTooLarge => {
                write!(f, "Value could not fit into a single pointer sized word.\nTry constructing a BoxValue instead.")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

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

impl<T> HasDrop for (DropFn, T) {
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

// Helper trait for calling trait functions that take bytes.
// Note that using SmallValue is currently only beneficial when the trait functions don't take
// it by value, as otherwise we would need to allocate a new Box.
pub trait GetBytesRef {
    fn get_bytes_ref(&self) -> &[u8];
}
pub trait GetBytesMut: GetBytesRef {
    fn get_bytes_mut(&mut self) -> &mut [u8];
}
pub trait GetBytes: GetBytesMut {
    fn get_bytes(self) -> Box<[u8]>;
}

impl GetBytesRef for Box<[u8]> {
    #[inline]
    fn get_bytes_ref(&self) -> &[u8] {
        &*self
    }
}
impl GetBytesMut for Box<[u8]> {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [u8] {
        &mut *self
    }
}
impl GetBytes for Box<[u8]> {
    #[inline]
    fn get_bytes(self) -> Box<[u8]> {
        self
    }
}

impl GetBytesRef for usize {
    #[inline]
    fn get_bytes_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}
impl GetBytesMut for usize {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}
impl GetBytes for usize {
    #[inline]
    fn get_bytes(self) -> Box<[u8]> {
        // This causes an additional allocation for every function call that accepts self by value.
        // TODO: Figure out how to eliminate this overhead.
        Bytes::box_into_box_bytes(Box::new(self))
    }
}

impl GetBytesRef for [u8] {
    #[inline]
    fn get_bytes_ref(&self) -> &[u8] {
        self
    }
}
impl GetBytesMut for [u8] {
    #[inline]
    fn get_bytes_mut(&mut self) -> &mut [u8] {
        self
    }
}

pub struct Value<B, V>
where
    B: GetBytesMut,
{
    pub(crate) bytes: ManuallyDrop<B>,
    pub(crate) type_id: TypeId,
    pub(crate) vtable: Arc<(DropFn, V)>,
}

//pub type SmallValue<V> = Value<usize, V>;
pub type BoxValue<V> = Value<Box<[u8]>, V>;
//pub type RcValue<V> = Value<Rc<[u8]>, V>;
//pub type ArcValue<V> = Value<Arc<[u8]>, V>;
/*
impl<V> SmallValue<V> {
    #[inline]
    pub fn small<T: Any + DropBytes>(value: T) -> Option<Value<usize, V>>
        where V: VTable<T>
    {
        value.try_into_usize().map(|usized_value| Value {
            bytes: ManuallyDrop::new(usized_value),
            type_id: TypeId::of::<T>(),
            vtable: Arc::new((T::drop_bytes, V::build_vtable())),
        })
    }
    /// Create a new `SmallValue` from a `usize` and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: usize,
        type_id: TypeId,
        vtable: Arc<(DropFn, V)>,
    ) -> Value<usize, V> {
        Value {
            bytes: ManuallyDrop::new(bytes),
            type_id,
            vtable,
        }
    }
}
*/

impl<V> BoxValue<V> {
    #[inline]
    pub fn new<T: Any + DropBytes>(value: T) -> Value<Box<[u8]>, V>
    where
        V: VTable<T>,
    {
        Value {
            bytes: ManuallyDrop::new(Bytes::box_into_box_bytes(Box::new(value))),
            type_id: TypeId::of::<T>(),
            vtable: Arc::new((T::drop_bytes, V::build_vtable())),
        }
    }
    /// Create a new `SmallValue` from boxed bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: Box<[u8]>,
        type_id: TypeId,
        vtable: Arc<(DropFn, V)>,
    ) -> Value<Box<[u8]>, V> {
        Value {
            bytes: ManuallyDrop::new(bytes),
            type_id,
            vtable,
        }
    }

    pub fn as_ref(&self) -> ValueRef<V> {
        ValueRef {
            bytes: &self.bytes,
            type_id: self.type_id,
            vtable: VTableRef::Ref(&self.vtable),
        }
    }

    pub fn as_mut(&mut self) -> ValueMut<V> {
        ValueMut {
            bytes: &mut self.bytes,
            type_id: self.type_id,
            vtable: VTableRef::Ref(&self.vtable),
        }
    }

    pub fn into_base<U: From<V>>(self) -> BoxValue<U>
    where
        V: Clone,
    {
        // Inhibit drop for self, it will be dropped by the returned value
        let md = ManuallyDrop::new(self);
        Value {
            bytes: md.bytes.clone(),
            type_id: md.type_id,
            vtable: Arc::new((md.vtable.0, U::from(md.vtable.1.clone()))),
        }
    }
}

impl<B: GetBytesMut, V: HasDebug> fmt::Debug for Value<B, V> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe { self.vtable.1.fmt_fn()(self.bytes.get_bytes_ref(), f) }
    }
}

impl<V: HasClone> Clone for Value<Box<[u8]>, V> {
    #[inline]
    fn clone(&self) -> Value<Box<[u8]>, V> {
        Value {
            bytes: ManuallyDrop::new(unsafe {
                self.vtable.1.clone_fn()(self.bytes.get_bytes_ref())
            }),
            type_id: self.type_id,
            vtable: Arc::clone(&self.vtable),
        }
    }
}

impl<B: GetBytesMut, V> Drop for Value<B, V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let mut bytes = ManuallyDrop::take(&mut self.bytes);
            self.vtable.drop_fn()(bytes.get_bytes_mut())
        }
    }
}

impl<B: GetBytesMut, V: HasPartialEq> PartialEq for Value<B, V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.vtable.1.eq_fn()(self.bytes.get_bytes_ref(), other.bytes.get_bytes_ref()) }
    }
}

impl<B: GetBytesMut, V: HasPartialEq> Eq for Value<B, V> {}

impl<B: GetBytesMut, V: HasHash> Hash for Value<B, V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            self.vtable.1.hash_fn()(self.bytes.get_bytes_ref(), state);
        }
    }
}

/// `VTable` defines a type that represents a virtual function table for some type `T`.
///
/// `T` is different than a type that can be turned into a trait object like `Box<dyn Any>`
/// because a `VTable` effectively decouples the type's behaviour from the data it contains.
///
/// This mechanism allows the virtual function table to be attached to a homogeneous container, to
/// prevent storing duplicates of these tables for each type instance stored in the container.
///
/// This is precisely how it is used to build `VecDyn<V>`, which is generic over the virtual table
/// rather than the type itself.
pub trait VTable<T> {
    fn build_vtable() -> Self;
}

impl<T: Copy> VTable<T> for () {
    fn build_vtable() -> Self {
        ()
    }
}

impl<B: GetBytes, V> Value<B, V> {
    impl_value_base!();
}

impl<V> Value<Box<[u8]>, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<Box<T>> {
        // Inhibit drop of self, since it will be dropped by the box.
        let mut s = ManuallyDrop::new(self);
        // This is safe since we check that self.bytes represent a `T`.
        if s.is::<T>() {
            Some(unsafe { Bytes::box_from_box_bytes(ManuallyDrop::take(&mut s.bytes)) })
        } else {
            None
        }
    }
}

impl<V> Value<usize, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<T> {
        let mut s = ManuallyDrop::new(self);
        // This is safe since we check that self.bytes represent a `T`.
        if s.is::<T>() {
            unsafe { Bytes::try_from_usize(ManuallyDrop::take(&mut s.bytes)) }
        } else {
            None
        }
    }
}

/// A VTable reference type.
///
/// Note we always need Drop because it's possible to clone ValueRef's contents, which need to know
/// how to drop themselves.
#[derive(Clone)]
pub(crate) enum VTableRef<'a, V> {
    Ref(&'a V),
    Owned(Box<V>),
}

impl<'a, V> AsRef<V> for VTableRef<'a, V> {
    #[inline]
    fn as_ref(&self) -> &V {
        match self {
            VTableRef::Ref(v) => v,
            VTableRef::Owned(v) => &*v,
        }
    }
}

/// A generic value reference into a buffer.
#[derive(Clone)]
pub struct ValueRef<'a, V> {
    pub(crate) bytes: &'a [u8],
    pub(crate) type_id: TypeId,
    pub(crate) vtable: VTableRef<'a, (DropFn, V)>,
}

impl<'a, V: HasHash> Hash for ValueRef<'a, V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            self.vtable.as_ref().hash_fn()(self.bytes.get_bytes_ref(), state);
        }
    }
}

impl<'a, V: HasDebug> fmt::Debug for ValueRef<'a, V> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe { self.vtable.as_ref().fmt_fn()(self.bytes.get_bytes_ref(), f) }
    }
}

impl<'a, V: HasPartialEq> PartialEq for ValueRef<'a, V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            self.vtable.as_ref().eq_fn()(self.bytes.get_bytes_ref(), other.bytes.get_bytes_ref())
        }
    }
}

impl<'a, V: HasEq> Eq for ValueRef<'a, V> {}

impl<'a, V> ValueRef<'a, V> {
    impl_value_base!();

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
            vtable: VTableRef::Owned(Box::new((T::drop_bytes, V::build_vtable()))),
        }
    }

    /// Create a new `ValueRef` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: &'a [u8],
        type_id: TypeId,
        vtable: &'a (DropFn, V),
    ) -> ValueRef<'a, V> {
        ValueRef {
            bytes,
            type_id,
            vtable: VTableRef::Ref(vtable),
        }
    }

    /// Clone the referenced value.
    ///
    /// Unlike the `Clone` trait, this function will produce an owned clone of the value pointed to
    /// by this value reference.
    #[inline]
    pub fn clone_value(&self) -> Value<Box<[u8]>, V>
    where
        V: HasClone + Clone,
    {
        Value {
            bytes: ManuallyDrop::new(unsafe { self.vtable.as_ref().clone_fn()(&self.bytes) }),
            type_id: self.type_id,
            vtable: Arc::from(self.vtable.as_ref().clone()),
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }

    pub fn into_base<U: From<V>>(&self) -> ValueRef<U>
    where
        V: Clone,
    {
        let vtable = self.vtable.as_ref();
        ValueRef {
            bytes: self.bytes,
            type_id: self.type_id,
            vtable: VTableRef::Owned(Box::new((vtable.0, U::from(vtable.1.clone())))),
        }
    }
}

/// A generic mutable value reference into a buffer.
pub struct ValueMut<'a, V> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) type_id: TypeId,
    pub(crate) vtable: VTableRef<'a, (DropFn, V)>,
}

impl<'a, V: HasDebug> fmt::Debug for ValueMut<'a, V> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe { self.vtable.as_ref().fmt_fn()(self.bytes, f) }
    }
}

impl<'a, V: HasPartialEq> PartialEq for ValueMut<'a, V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            self.vtable.as_ref().eq_fn()(self.bytes.get_bytes_ref(), other.bytes.get_bytes_ref())
        }
    }
}

impl<'a, V: HasEq> Eq for ValueMut<'a, V> {}

impl<'a, V> ValueMut<'a, V> {
    impl_value_base!();

    /// Create a new `ValueRef` from a typed reference.
    #[inline]
    pub fn new<T: Any>(typed: &'a mut T) -> ValueMut<'a, V>
    where
        V: VTable<T>,
    {
        ValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
            vtable: VTableRef::Owned(Box::new((T::drop_bytes, V::build_vtable()))),
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap<'b>(&mut self, other: &mut ValueMut<'b, V>) {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.swap_with_slice(other.bytes);
        }
    }

    /// Clone `other` into `self`.
    ///
    /// This function will call `drop` on any values stored in `self`.
    #[inline]
    pub fn clone_from<'b>(&mut self, other: impl Into<ValueRef<'b, V>>)
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
        }
    }

    /// Create a new `ValueMut` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: &'a mut [u8],
        type_id: TypeId,
        vtable: &'a (DropFn, V),
    ) -> ValueMut<'a, V> {
        ValueMut {
            bytes,
            type_id,
            vtable: VTableRef::Ref(vtable),
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
    }

    //pub fn into_base<U: From<V>>(self) -> ValueMut<'a, U> {
    //    ValueMut {
    //        bytes: self.bytes,
    //        type_id: self.type_id,
    //        vtable: VTableRef::Owned(Box::new((self.vtable.as_ref().0, U::from(&self.vtable.as_ref().1)))),
    //    }
    //}
}

/// A generic value reference to a `Copy` type.
#[derive(Copy, Clone, Debug)]
pub struct CopyValueRef<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) type_id: TypeId,
}

impl<'a> CopyValueRef<'a> {
    impl_value_base!();

    /// Create a new `CopyValueRef` from a typed reference.
    #[inline]
    pub fn new<T: Elem>(typed: &'a T) -> CopyValueRef<'a> {
        CopyValueRef {
            bytes: typed.as_bytes(),
            type_id: TypeId::of::<T>(),
        }
    }

    /// Create a new `CopyValueref` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(bytes: &'a [u8], type_id: TypeId) -> CopyValueRef<'a> {
        CopyValueRef { bytes, type_id }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Elem>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }
}

/// A generic mutable `Copy` value reference.
#[derive(Debug)]
pub struct CopyValueMut<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) type_id: TypeId,
}

impl<'a> CopyValueMut<'a> {
    impl_value_base!();

    /// Create a new `CopyValueMut` from a typed mutable reference.
    #[inline]
    pub fn new<T: Elem>(typed: &'a mut T) -> CopyValueMut<'a> {
        CopyValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
        }
    }

    /// Create a new `CopyValueMut` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(bytes: &'a mut [u8], type_id: TypeId) -> CopyValueMut<'a> {
        CopyValueMut { bytes, type_id }
    }

    /// Copy value from `other` to `self` and return `Self`.
    ///
    /// This function returns `None` if the values have different types.
    #[inline]
    pub fn copy(self, other: CopyValueRef<'a>) -> Option<Self> {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.copy_from_slice(other.bytes);
            Some(self)
        } else {
            None
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap(&mut self, other: &mut CopyValueMut) {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.swap_with_slice(&mut other.bytes);
            std::mem::swap(&mut self.type_id, &mut other.type_id);
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Elem>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
    }
}

/*
 * Valid conversions.
 */

impl<'a, V> From<ValueMut<'a, V>> for ValueRef<'a, V> {
    #[inline]
    fn from(v: ValueMut<'a, V>) -> ValueRef<'a, V> {
        ValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
            vtable: v.vtable,
        }
    }
}

impl<'a> From<CopyValueMut<'a>> for CopyValueRef<'a> {
    #[inline]
    fn from(v: CopyValueMut<'a>) -> CopyValueRef<'a> {
        CopyValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
        }
    }
}
