//! This module defines a mechanism for type earasure simplar to `dyn Any`.
//!
//! This method allows storing and referencing values coming from a `DataBuffer` in other `std`
//! containers without needing a downcast.
//!

use std::any::{Any, TypeId};
use std::rc::Rc;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::fmt;

use crate::vec_clone::Elem;
use crate::traits::*;
use crate::bytes::*;

// Implement the basis for all value types.
macro_rules! impl_value_base {
    () => {
        /// Get the size of the value pointed-to by this reference.
        #[inline]
        pub fn size(&self) -> usize {
            self.bytes.as_ref().len()
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

pub(crate) trait HasDrop {
    fn drop_fn(&self) -> &DropFn;
}

pub trait HasClone {
    fn clone_fn(&self) -> &CloneFnType;
    fn clone_from_fn(&self) -> &CloneFromFnType;
}

pub trait HasHash {
    fn hash_fn(&self) -> &HashFnType;
}

pub trait HasPartialEq {
    fn eq_fn(&self) -> &EqFnType;
}

pub trait HasDebug {
    fn fmt_fn(&self) -> &FmtFnType;
}

impl<T> HasDrop for (DropFn, T) {
    #[inline]
    fn drop_fn(&self) -> &DropFn { &self.0 }
}

// Helper trait for dropping bytes in pointer containers.
pub trait GetBytesMut {
    fn get_bytes_mut(&mut self) -> Option<&mut [u8]>;
}

impl GetBytesMut for Box<[u8]> {
    fn get_bytes_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut *self)
    }
}

impl GetBytesMut for Rc<[u8]> {
    fn get_bytes_mut(&mut self) -> Option<&mut [u8]> {
        Rc::get_mut(self)
    }
}

impl GetBytesMut for Arc<[u8]> {
    fn get_bytes_mut(&mut self) -> Option<&mut [u8]> {
        Arc::get_mut(self)
    }
}

pub struct DynValue<B, V> where B: GetBytesMut {
    pub(crate) bytes: ManuallyDrop<B>,
    pub(crate) type_id: TypeId,
    pub(crate) vtable: Arc<(DropFn, V)>,
}

//pub type PtrDynValue<V> = DynValue<usize, V>;
//pub type BoxDynValue<V> = DynValue<Box<[u8]>, V>;
//pub type RcDynValue<V> = DynValue<Rc<[u8]>, V>;
//pub type ArcDynValue<V> = DynValue<Arc<[u8]>, V>;

impl<B: GetBytesMut, V> DynValue<B, V> {
    #[inline]
    pub fn new<T: Any + IntoRaw<B> + DropBytes + Dyn<VTable = V>>(value: T) -> DynValue<B, V> {
        DynValue {
            bytes: ManuallyDrop::new(value.into_raw()),
            type_id: TypeId::of::<T>(),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
        }
    }
}
//impl<V> BoxDynValue<V> {
//    #[inline]
//    pub fn new<T: Any + DropBytes + Dyn<VTable = V>>(value: Box<T>) -> DynValue<Box<[u8]>, V> {
//        DynValue {
//            bytes: ManuallyDrop::new(Bytes::box_into_box_bytes(value)),
//            type_id: TypeId::of::<T>(),
//            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
//        }
//    }
//}
//impl<V> RcDynValue<V> {
//    #[inline]
//    pub fn new<T: Any + DropBytes + Dyn<VTable = V>>(value: Rc<T>) -> DynValue<Rc<[u8]>, V> {
//        DynValue {
//            bytes: ManuallyDrop::new(Bytes::rc_into_rc_bytes(value)),
//            type_id: TypeId::of::<T>(),
//            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
//        }
//    }
//}
//impl<V> ArcDynValue<V> {
//    #[inline]
//    pub fn new<T: Any + DropBytes + Dyn<VTable = V>>(value: Arc<T>) -> DynValue<Arc<[u8]>, V> {
//        DynValue {
//            bytes: ManuallyDrop::new(Bytes::arc_into_arc_bytes(value)),
//            type_id: TypeId::of::<T>(),
//            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
//        }
//    }
//}

impl<B: GetBytesMut + AsRef<[u8]>, V: HasDebug> fmt::Debug for DynValue<B, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            self.vtable.1.fmt_fn()(self.bytes.as_ref(), f)
        }
    }
}


impl<V: HasClone> Clone for DynValue<Box<[u8]>, V> {
    #[inline]
    fn clone(&self) -> DynValue<Box<[u8]>, V> {
        DynValue {
            bytes: ManuallyDrop::new(unsafe { self.vtable.1.clone_fn()(self.bytes.as_ref()) }),
            type_id: self.type_id,
            vtable: Arc::clone(&self.vtable),
        }
    }
}

impl<V> Clone for DynValue<Rc<[u8]>, V> {
    #[inline]
    fn clone(&self) -> DynValue<Rc<[u8]>, V> {
        DynValue {
            bytes: ManuallyDrop::new(Rc::clone(&self.bytes)),
            type_id: self.type_id,
            vtable: Arc::clone(&self.vtable),
        }
    }
}

impl<V> Clone for DynValue<Arc<[u8]>, V> {
    #[inline]
    fn clone(&self) -> DynValue<Arc<[u8]>, V> {
        DynValue {
            bytes: ManuallyDrop::new(Arc::clone(&self.bytes)),
            type_id: self.type_id,
            vtable: Arc::clone(&self.vtable),
        }
    }
}

impl<B: GetBytesMut, V> Drop for DynValue<B, V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // If we have unique access, drop the contents.
            if let Some(bytes) = self.bytes.get_bytes_mut() {
                self.vtable.drop_fn().0(bytes)
            }
            // Now drop the Rc<[u8]>. This is safe because self will not be used after this.
            let _ = ManuallyDrop::take(&mut self.bytes);
        }
    }
}

impl<B: GetBytesMut + AsRef<[u8]>, V: HasPartialEq> PartialEq for DynValue<B, V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            self.vtable.1.eq_fn()(self.bytes.as_ref(), other.bytes.as_ref())
        }
    }
}

impl<B: GetBytesMut + AsRef<[u8]>, V: HasPartialEq> Eq for DynValue<B, V> { }

impl<B: GetBytesMut + AsRef<[u8]>, V: HasHash> Hash for DynValue<B, V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            self.vtable.1.hash_fn()(self.bytes.as_ref(), state);
        }
    }
}

/// `Dyn` defines a type that can be converted into a virtual function table.
///
/// This is different than a type that can be turned into a trait object like `Box<dyn Any>`
/// because it decouples the type's behavior from the data it contains.
///
/// This mechanism allows the virtual function table to be attached to a homogeneous container, to
/// prevent storing duplicates of these tables for each type instance stored in the container.
///
/// This is precisely how it is used to build `VecDyn<V>`, which is generic over the virtual table
/// rather than the type itself.
pub trait Dyn {
    type VTable;
    fn build_vtable() -> Self::VTable;
}


impl<B: GetBytesMut + AsRef<[u8]>, V> DynValue<B, V> {
    impl_value_base!();
}

impl<V> DynValue<Box<[u8]>, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<Box<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut s| unsafe {
            Bytes::box_from_box_bytes(ManuallyDrop::take(&mut s.bytes))
        })
    }
}

impl<V> DynValue<Rc<[u8]>, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<Rc<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe { Bytes::rc_from_rc_bytes(ManuallyDrop::take(&mut b.bytes)) })
    }
}

impl<V> DynValue<Arc<[u8]>, V> {
    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<Arc<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe { Bytes::arc_from_arc_bytes(ManuallyDrop::take(&mut b.bytes)) })
    }
}

/// A generic value reference into a buffer.
#[derive(Clone)]
pub struct DynValueRef<'a, V> {
    pub(crate) bytes: &'a [u8],
    pub(crate) type_id: TypeId,
    pub(crate) vtable: Arc<(DropFn, V)>,
}

impl<'a, V: HasDebug> fmt::Debug for DynValueRef<'a, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            self.vtable.1.fmt_fn()(self.bytes, f)
        }
    }
}

impl<'a, V> DynValueRef<'a, V> {
    impl_value_base!();
    
    /// Create a new `ValueRef` from a typed reference.
    #[inline]
    pub fn new<T: Any + DropBytes + Dyn<VTable = V>>(typed: &'a T) -> DynValueRef<'a, V> {
        DynValueRef {
            bytes: typed.as_bytes(),
            type_id: TypeId::of::<T>(),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
        }
    }

    /// Create a new `ValueRef` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(bytes: &'a [u8], type_id: TypeId, vtable: Arc<(DropFn, V)>) -> DynValueRef<'a, V> {
        DynValueRef { bytes, type_id, vtable }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Bytes + 'static>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }
}

#[derive(Debug)]
pub struct BoxValue {
    pub(crate) bytes: ManuallyDrop<Box<[u8]>>,
    pub(crate) type_id: TypeId,
    pub(crate) clone_fn: CloneFn,
    pub(crate) drop_fn: DropFn,
}

impl Clone for BoxValue {
    fn clone(&self) -> BoxValue {
        BoxValue {
            bytes: ManuallyDrop::new(unsafe { self.clone_fn.0(self.bytes.as_ref()) }),
            type_id: self.type_id,
            clone_fn: self.clone_fn,
            drop_fn: self.drop_fn,
        }
    }
}

impl Drop for BoxValue {
    fn drop(&mut self) {
        unsafe {
            self.drop_fn.0(&mut *self.bytes);
        }
    }
}

impl BoxValue {
    impl_value_base!();

    #[inline]
    pub fn new<T: Elem>(typed: Box<T>) -> BoxValue {
        BoxValue {
            bytes: ManuallyDrop::new(Bytes::box_into_box_bytes(typed)),
            type_id: TypeId::of::<T>(),
            clone_fn: CloneFn(T::clone_bytes),
            drop_fn: DropFn(T::drop_bytes),
        }
    }

    //#[inline]
    //pub(crate) unsafe fn clone_from_bytes(bytes: &[u8], type_id: TypeId, clone_fn: CloneFn, drop_fn: DropFn) -> BoxValue {
    //    debug_assert_eq!(bytes.len(), std::mem::size_of::<usize>());
    //    BoxValue {
    //        bytes: clone_fn.0(bytes),
    //        type_id,
    //        clone_fn,
    //        drop_fn,
    //    }
    //}

    /// Downcast this value reference into a boxed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Elem>(self) -> Option<Box<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe { Bytes::box_from_box_bytes(ManuallyDrop::take(&mut b.bytes)) })
    }
}

#[derive(Clone, Debug)]
pub struct RcValue {
    pub(crate) bytes: ManuallyDrop<Rc<[u8]>>,
    pub(crate) type_id: TypeId,
    pub(crate) drop_fn: DropFn,
    // We don't need a clone function here since cloning an Rc is independent of the value it
    // contains.
}

impl Drop for RcValue {
    fn drop(&mut self) {
        unsafe {
            // If we have unique access, drop the contents.
            if let Some(bytes) = Rc::get_mut(&mut self.bytes) {
                self.drop_fn.0(bytes)
            }
            // Now drop the Rc<[u8]>. This is safe because self will not be used after this.
            let _ = ManuallyDrop::take(&mut self.bytes);
        }
    }
}

impl RcValue {
    impl_value_base!();

    #[inline]
    pub fn new<T: Bytes + DropBytes>(typed: Rc<T>) -> RcValue {
        RcValue {
            bytes: ManuallyDrop::new(Bytes::rc_into_rc_bytes(typed)),
            type_id: TypeId::of::<T>(),
            drop_fn: DropFn(T::drop_bytes),
        }
    }

    //#[inline]
    //pub(crate) unsafe fn clone_from_bytes(bytes: &[u8], type_id: TypeId, drop_fn: DropFn) -> RcValue {
    //    debug_assert_eq!(bytes.len(), std::mem::size_of::<usize>());
    //    RcValue {
    //        bytes: Rc::clone(Rc::<[u8]>::from_bytes(bytes)),
    //        type_id,
    //        drop_fn,
    //    }
    //}

    /// Downcast this value reference into an `Rc<T>` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Bytes>(self) -> Option<Rc<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe { Bytes::rc_from_rc_bytes(ManuallyDrop::take(&mut b.bytes)) })
    }
}

impl<T: Bytes + 'static> From<Rc<T>> for RcValue {
    #[inline]
    fn from(rc: Rc<T>) -> RcValue {
        RcValue::new(rc)
    }
}

#[derive(Clone, Debug)]
pub struct ArcValue {
    pub(crate) bytes: ManuallyDrop<Arc<[u8]>>,
    pub(crate) type_id: TypeId,
    pub(crate) drop_fn: DropFn,
}

impl Drop for ArcValue {
    fn drop(&mut self) {
        unsafe {
            // If we have unique access, drop the contents.
            if let Some(bytes) = Arc::get_mut(&mut self.bytes) {
                self.drop_fn.0(bytes)
            }
            // Now drop the Arc<[u8]>. This is safe because self will not be used after this.
            let _ = ManuallyDrop::take(&mut self.bytes);
        }
    }
}

impl ArcValue {
    impl_value_base!();

    #[inline]
    pub fn new<T: Bytes>(typed: Arc<T>) -> ArcValue {
        ArcValue {
            bytes: ManuallyDrop::new(Bytes::arc_into_arc_bytes(typed)),
            type_id: TypeId::of::<T>(),
            drop_fn: DropFn(Arc::<T>::drop_bytes),
        }
    }

    //#[inline]
    //pub(crate) unsafe fn clone_from_bytes(bytes: &[u8], type_id: TypeId, drop_fn: DropFn) -> ArcValue {
    //    debug_assert_eq!(bytes.len(), std::mem::size_of::<usize>());
    //    ArcValue {
    //        bytes: Arc::clone(Arc::<[u8]>::from_bytes(bytes)),
    //        type_id,
    //        drop_fn,
    //    }
    //}

    /// Downcast this value reference into an `Arc<T>` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Bytes>(self) -> Option<Arc<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe { Bytes::arc_from_arc_bytes(ManuallyDrop::take(&mut b.bytes)) })
    }
}

impl<T: Bytes + 'static> From<Arc<T>> for ArcValue {
    #[inline]
    fn from(arc: Arc<T>) -> ArcValue {
        ArcValue::new(arc)
    }
}

/// A generic value reference into a buffer.
#[derive(Copy, Clone, Debug)]
pub struct ValueRef<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) type_id: TypeId,
}

impl<'a> ValueRef<'a> {
    impl_value_base!();
    
    /// Create a new `ValueRef` from a typed reference.
    #[inline]
    pub fn new<T: Elem>(typed: &'a T) -> ValueRef<'a> {
        ValueRef {
            bytes: typed.as_bytes(),
            type_id: TypeId::of::<T>(),
        }
    }

    /// Create a new `ValueRef` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(bytes: &'a [u8], type_id: TypeId) -> ValueRef<'a> {
        ValueRef { bytes, type_id }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Bytes + 'static>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }
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
    pub fn new<T: Bytes + Copy + 'static>(typed: &'a T) -> CopyValueRef<'a> {
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
    pub fn downcast<T: Bytes + Copy + 'static>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }
}

/// A generic mutable value reference into a buffer.
#[derive(Debug)]
pub struct ValueMut<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) type_id: TypeId,
    pub(crate) clone_from_fn: CloneFromFn,
}

impl<'a> ValueMut<'a> {
    impl_value_base!();

    /// Create a new `ValueMut` from a typed mutable reference.
    #[inline]
    pub fn new<T: Elem>(typed: &'a mut T) -> ValueMut<'a> {
        ValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
            clone_from_fn: CloneFromFn(T::clone_from_bytes),
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
        clone_from_fn: CloneFromFnType,
    ) -> ValueMut<'a> {
        ValueMut {
            bytes,
            type_id,
            clone_from_fn: CloneFromFn(clone_from_fn),
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap(&mut self, other: &mut ValueMut) {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.swap_with_slice(other.bytes);
        }
    }

    /// Clone `other` into `self`.
    ///
    /// This function will call `drop` on any values stored in `self`.
    #[inline]
    pub fn clone_from(&mut self, other: impl Into<ValueRef<'a>>) {
        let other = other.into();
        if self.value_type_id() == other.value_type_id() {
            unsafe {
                // We are cloning other.bytes into self.bytes.
                // This function will call the appropriate typed clone_from function, which will
                // drop the previous value of self.bytes.
                self.clone_from_fn.0(&mut self.bytes, other.bytes);
            }
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Elem>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
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
    pub fn new<T: Bytes + Copy + 'static>(typed: &'a mut T) -> CopyValueMut<'a> {
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
    pub fn downcast<T: Bytes + Copy + 'static>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
    }
}

/*
 * Valid conversions.
 */

impl<'a> From<ValueMut<'a>> for ValueRef<'a> {
    #[inline]
    fn from(v: ValueMut<'a>) -> ValueRef<'a> {
        ValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
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
