//! This module defines a mechanism for type earasure simplar to `dyn Any`.
//!
//! This method allows storing and referencing values coming from a `DataBuffer` in other `std`
//! containers without needing a downcast.
//!

use std::any::TypeId;
use std::rc::Rc;
use std::sync::Arc;

use crate::vec_clone::{DropBytes, CloneFromFn, CloneFn, DropFn, Elem};
use crate::bytes::*;


// Implement the basis for all value types.
macro_rules! impl_value_base {
    ($value:ty, $bytes_type:ty) => {
        /// Get the size of the value pointed-to by this reference.
        #[inline]
        pub fn size(&self) -> usize {
            self.bytes.len()
        }

        /// Get the `TypeId` of the referenced value.
        #[inline]
        pub fn type_id(&self) -> TypeId {
            self.type_id
        }

        /// Returns `true` if this referenced value's type is the same as `T`.
        #[inline]
        pub fn is<T: 'static>(&self) -> bool {
            self.type_id() == TypeId::of::<T>()
        }

        // Check that this value represents the given type, and if so return the bytes.
        // This is a helper for downcasts.
        #[inline]
        fn check<T: 'static>(self) -> Option<$bytes_type> {
            if self.is::<T>() {
                Some(self.bytes)
            } else {
                None
            }
        }
    };
}

#[derive(Clone, Debug)]
pub struct BoxValue {
    pub(crate) bytes: Box<[u8]>,
    pub(crate) type_id: TypeId,
    pub(crate) clone_fn: CloneFn,
    pub(crate) drop_fn: DropFn,
}

impl BoxValue {
    impl_value_base!(BoxValue, Box<[u8]>);

    #[inline]
    pub fn new<T: Elem>(typed: Box<T>) -> BoxValue {
        BoxValue {
            bytes: Bytes::box_into_box_bytes(typed),
            type_id: TypeId::of::<T>(),
            clone_fn: CloneFn::new(T::clone_bytes),
            drop_fn: DropFn::new(T::drop_bytes),
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
        self.check::<T>().map(|b| unsafe { Bytes::box_from_box_bytes(b) })
    }
}

#[derive(Clone, Debug)]
pub struct RcValue {
    pub(crate) bytes: Rc<[u8]>,
    pub(crate) type_id: TypeId,
    pub(crate) drop_fn: DropFn,
    // We don't need a clone function here since cloning an Rc is independent of the value it
    // contains.
}

impl RcValue {
    impl_value_base!(RcValue, Rc<[u8]>);

    #[inline]
    pub fn new<T: Bytes>(typed: Rc<T>) -> RcValue {
        RcValue {
            bytes: Bytes::rc_into_rc_bytes(typed),
            type_id: TypeId::of::<T>(),
            drop_fn: DropFn::new(Rc::<T>::drop_bytes),
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
        self.check::<T>().map(|b| unsafe { Bytes::rc_from_rc_bytes(b) })
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
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) type_id: TypeId,
    pub(crate) drop_fn: DropFn,
}

impl ArcValue {
    impl_value_base!(ArcValue, Arc<[u8]>);

    #[inline]
    pub fn new<T: Bytes>(typed: Arc<T>) -> ArcValue {
        ArcValue {
            bytes: Bytes::arc_into_arc_bytes(typed),
            type_id: TypeId::of::<T>(),
            drop_fn: DropFn::new(Arc::<T>::drop_bytes),
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
        self.check::<T>().map(|b| unsafe { Bytes::arc_from_arc_bytes(b) })
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
    impl_value_base!(ValueRef, &'a [u8]);
    
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
        self.check::<T>().map(|b| unsafe { Bytes::from_bytes(b) })
    }
}

/// A generic value reference to a `Copy` type.
#[derive(Copy, Clone, Debug)]
pub struct CopyValueRef<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) type_id: TypeId,
}

impl<'a> CopyValueRef<'a> {
    impl_value_base!(CopyValueRef, &'a [u8]);

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
        self.check::<T>().map(|b| unsafe { Bytes::from_bytes(b) })
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
    impl_value_base!(ValueMut,  &'a mut [u8]);

    /// Create a new `ValueMut` from a typed mutable reference.
    #[inline]
    pub fn new<T: Elem>(typed: &'a mut T) -> ValueMut<'a> {
        ValueMut {
            bytes: typed.as_bytes_mut(),
            type_id: TypeId::of::<T>(),
            clone_from_fn: CloneFromFn::new(T::clone_from_bytes),
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
        clone_from_fn: CloneFromFn,
    ) -> ValueMut<'a> {
        ValueMut {
            bytes,
            type_id,
            clone_from_fn,
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap(&mut self, other: &mut ValueMut) {
        if self.type_id() == other.type_id() {
            self.bytes.swap_with_slice(other.bytes);
        }
    }

    /// Clone `other` into `self`.
    ///
    /// This function will call `drop` on any values stored in `self`.
    #[inline]
    pub fn clone_from(&mut self, other: impl Into<ValueRef<'a>>) {
        let other = other.into();
        if self.type_id() == other.type_id() {
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
        self.check::<T>()
            .map(|b| unsafe { Bytes::from_bytes_mut(b) })
    }
}

/// A generic mutable `Copy` value reference.
#[derive(Debug)]
pub struct CopyValueMut<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) type_id: TypeId,
}

impl<'a> CopyValueMut<'a> {
    impl_value_base!(CopyValueMut,  &'a mut [u8]);

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
        if self.type_id() == other.type_id() {
            self.bytes.copy_from_slice(other.bytes);
            Some(self)
        } else {
            None
        }
    }

    /// Swap the values between `other` and `self`.
    #[inline]
    pub fn swap(&mut self, other: &mut CopyValueMut) {
        if self.type_id() == other.type_id() {
            self.bytes.swap_with_slice(&mut other.bytes);
            std::mem::swap(&mut self.type_id, &mut other.type_id);
        }
    }

    /// Downcast this value reference into a borrowed `T` type. Return `None` if the downcast fails.
    #[inline]
    pub fn downcast<T: Bytes + Copy + 'static>(self) -> Option<&'a mut T> {
        // This is safe since we check that self.bytes represent a `T`.
        self.check::<T>()
            .map(|b| unsafe { Bytes::from_bytes_mut(b) })
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
