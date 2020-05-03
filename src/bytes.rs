//! This module deals with conversions of types between bytes.
//! For reference on potential pitfals here are two articles:
//! - [Safe trasmute proposal](https://internals.rust-lang.org/t/pre-rfc-safe-transmute/11347)
//! - [Notes on Type Layouts aand ABIs in Rust](https://gankra.github.io/blah/rust-layouts-and-abis/)
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;
use std::slice;

/// A trait defining a safe way to convert to and from byte pointers and slices.
pub trait Bytes
where
    Self: Sized + 'static,
{
    /// Get a mut pointer to the bytes representing `Self`.
    #[inline]
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self as *mut Self as *mut u8
    }

    /// Get a const pointer to the bytes representing `Self`.
    #[inline]
    fn as_byte_ptr(&self) -> *const u8 {
        self as *const Self as *const u8
    }

    /// Construct `Self` from a mut pointer to bytes.
    ///
    /// # Safety
    ///
    /// This is inherently unsafe, especially for pointer types since `bytes` can store arbitrary
    /// data. In general, this `fn` is only required to guarantee safety for `bytes` returned by
    /// the `as_byte_mut_ptr` function.
    #[inline]
    unsafe fn from_byte_mut_ptr<'a>(bytes: *mut u8) -> &'a mut Self {
        &mut *(bytes as *mut Self)
    }

    /// Construct `Self` from a const pointer to bytes.
    ///
    /// # Safety
    ///
    /// This is inherently unsafe, especially for pointer types since `bytes` can store arbitrary
    /// data. In general, this `fn` is only required to guarantee safety for `bytes` returned by
    /// the `as_byte_ptr` function.
    #[inline]
    unsafe fn from_byte_ptr<'a>(bytes: *const u8) -> &'a Self {
        &*(bytes as *const Self)
    }

    /// Get a mutable slice of bytes representing `Self`.
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { slice::from_raw_parts_mut(self.as_byte_mut_ptr(), size_of::<Self>()) }
    }

    /// Get a slice of bytes representing `Self`.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { slice::from_raw_parts(self.as_byte_ptr(), size_of::<Self>()) }
    }

    /// Construct a mutable reference to `Self` from a mutable slice of bytes.
    ///
    /// In contrast to the `from_byte_mut_ptr` function, this function does an additional size
    /// check to ensure that the given slice has the correct size. This is not enough to guarantee
    /// safety, but prevents out-of-bounds reads.
    ///
    /// # Safety
    ///
    /// This is inherently unsafe, especially for pointer types since `bytes` can store arbitrary
    /// data. In general, this `fn` is only required to guarantee safety for `bytes` returned by
    /// the `as_bytes` function.
    #[inline]
    unsafe fn from_bytes_mut(bytes: &mut [u8]) -> &mut Self {
        assert_eq!(bytes.len(), size_of::<Self>());
        Self::from_byte_mut_ptr(bytes.as_mut_ptr())
    }

    /// Construct `Self` from a slice of bytes.
    ///
    /// In contrast to the `from_byte_ptr` function, this function does an additional size check
    /// to ensure that the given slice has the correct size. This is not enough to
    /// guarantee safety, but prevents out-of-bounds reads.
    ///
    /// # Safety
    ///
    /// This is inherently unsafe, especially for pointer types since `bytes` can store arbitrary
    /// data. In general, this `fn` is only required to guarantee safety for `bytes` returned by
    /// the `as_bytes` function.
    #[inline]
    unsafe fn from_bytes(bytes: &[u8]) -> &Self {
        assert_eq!(bytes.len(), size_of::<Self>());
        Self::from_byte_ptr(bytes.as_ptr())
    }

    #[inline]
    fn rc_into_rc_bytes(rc: Rc<Self>) -> Rc<[u8]> {
        let byte_ptr = Rc::into_raw(rc) as *const u8;
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { Rc::from_raw(slice::from_raw_parts(byte_ptr, size_of::<Self>())) }
    }

    #[inline]
    unsafe fn rc_from_rc_bytes(rc: Rc<[u8]>) -> Rc<Self> {
        Rc::from_raw(Rc::into_raw(rc) as *const Self)
    }

    #[inline]
    fn arc_into_arc_bytes(arc: Arc<Self>) -> Arc<[u8]> {
        let byte_ptr = Arc::into_raw(arc) as *const u8;
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { Arc::from_raw(slice::from_raw_parts(byte_ptr, size_of::<Self>())) }
    }

    #[inline]
    unsafe fn arc_from_arc_bytes(arc: Arc<[u8]>) -> Arc<Self> {
        Arc::from_raw(Arc::into_raw(arc) as *const Self)
    }

    #[inline]
    fn box_into_box_bytes(b: Box<Self>) -> Box<[u8]> {
        let byte_ptr = Box::into_raw(b) as *mut u8;
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { Box::from_raw(slice::from_raw_parts_mut(byte_ptr, size_of::<Self>())) }
    }

    #[inline]
    unsafe fn box_from_box_bytes(b: Box<[u8]>) -> Box<Self> {
        Box::from_raw(Box::into_raw(b) as *mut Self)
    }
}

// This is of course wilidly unsafe, but it makes the tests compile for now.
impl<T> Bytes for T where T: 'static {}

pub unsafe trait IntoRaw<T>: Sized {
    /// Performs conversion into a raw type `T`.
    fn into_raw(self) -> T;
}

pub unsafe trait FromRaw<T>: Sized {
    /// Performs conversion from a raw type `T` into `Self`.
    fn from_raw(_: T) -> Self;
}
