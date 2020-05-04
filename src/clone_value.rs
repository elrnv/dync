//! This module defines a mechanism for type earasure simplar to `dyn Any`.
//!
//! This method allows storing and referencing values coming from a `DataBuffer` in other `std`
//! containers without needing a downcast.
//!
#![allow(dead_code)]

use std::any::TypeId;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::Arc;

use crate::bytes::*;
use crate::traits::*;
use crate::value::GetBytesRef;
use crate::vec_clone::Elem as CloneElem;

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
    pub fn new<T: CloneElem>(typed: Box<T>) -> BoxValue {
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
    pub fn downcast<T: CloneElem>(self) -> Option<Box<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe {
            Bytes::box_from_box_bytes(ManuallyDrop::take(&mut b.bytes))
        })
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
    pub fn new<T: DropBytes + 'static>(typed: Rc<T>) -> RcValue {
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
    pub fn downcast<T: 'static>(self) -> Option<Rc<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe {
            Bytes::rc_from_rc_bytes(ManuallyDrop::take(&mut b.bytes))
        })
    }
}

impl<T: 'static> From<Rc<T>> for RcValue {
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
    pub fn new<T: 'static>(typed: Arc<T>) -> ArcValue {
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
    pub fn downcast<T: 'static>(self) -> Option<Arc<T>> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|mut b| unsafe {
            Bytes::arc_from_arc_bytes(ManuallyDrop::take(&mut b.bytes))
        })
    }
}

impl<T: 'static> From<Arc<T>> for ArcValue {
    #[inline]
    fn from(arc: Arc<T>) -> ArcValue {
        ArcValue::new(arc)
    }
}

/// A generic value reference into a buffer.
#[derive(Copy, Clone, Debug)]
pub struct CloneValueRef<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) type_id: TypeId,
}

impl<'a> CloneValueRef<'a> {
    impl_value_base!();

    /// Create a new `CloneValueRef` from a typed reference.
    #[inline]
    pub fn new<T: CloneElem>(typed: &'a T) -> CloneValueRef<'a> {
        CloneValueRef {
            bytes: typed.as_bytes(),
            type_id: TypeId::of::<T>(),
        }
    }

    /// Create a new `CloneValueRef` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(bytes: &'a [u8], type_id: TypeId) -> CloneValueRef<'a> {
        CloneValueRef { bytes, type_id }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: 'static>(self) -> Option<&'a T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes(b.bytes) })
    }
}

/// A generic mutable value reference into a buffer.
#[derive(Debug)]
pub struct CloneValueMut<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) type_id: TypeId,
    pub(crate) clone_from_fn: CloneFromFn,
}

impl<'a> CloneValueMut<'a> {
    impl_value_base!();

    /// Create a new `CloneValueMut` from a typed mutable reference.
    #[inline]
    pub fn new<T: CloneElem>(typed: &'a mut T) -> CloneValueMut<'a> {
        CloneValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
            clone_from_fn: CloneFromFn(T::clone_from_bytes),
        }
    }

    /// Create a new `CloneValueMut` from a slice of bytes and an associated `TypeId`.
    ///
    /// # Safety
    ///
    /// The given bytes must be the correct representation of the type given `TypeId`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        bytes: &'a mut [u8],
        type_id: TypeId,
        clone_from_fn: CloneFromFnType,
    ) -> CloneValueMut<'a> {
        CloneValueMut {
            bytes,
            type_id,
            clone_from_fn: CloneFromFn(clone_from_fn),
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap(&mut self, other: &mut CloneValueMut) {
        if self.value_type_id() == other.value_type_id() {
            self.bytes.swap_with_slice(other.bytes);
        }
    }

    /// Clone `other` into `self`.
    ///
    /// This function will call `drop` on any values stored in `self`.
    #[inline]
    pub fn clone_from(&mut self, other: impl Into<CloneValueRef<'a>>) {
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
    pub fn downcast<T: CloneElem>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.downcast_with::<T, _, _>(|b| unsafe { Bytes::from_bytes_mut(b.bytes) })
    }
}

/*
 * Valid conversions.
 */

impl<'a> From<CloneValueMut<'a>> for CloneValueRef<'a> {
    #[inline]
    fn from(v: CloneValueMut<'a>) -> CloneValueRef<'a> {
        CloneValueRef {
            bytes: v.bytes,
            type_id: v.type_id,
        }
    }
}
