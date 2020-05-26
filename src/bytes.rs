//! This module deals with conversions of types between bytes.
//! For reference on potential pitfals here are two articles:
//! - [Safe trasmute proposal](https://internals.rust-lang.org/t/pre-rfc-safe-transmute/11347)
//! - [Notes on Type Layouts aand ABIs in Rust](https://gankra.github.io/blah/rust-layouts-and-abis/)
use std::mem::size_of;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::slice;
use std::sync::Arc;

/// A trait defining a safe way to convert to and from byte pointers and slices.
pub(crate) trait Bytes
where
    Self: Sized + 'static,
{
    /// Get a mut pointer to the bytes representing `Self`.
    #[inline]
    fn as_byte_mut_ptr(&mut self) -> *mut MaybeUninit<u8> {
        self as *mut Self as *mut MaybeUninit<u8>
    }

    /// Get a const pointer to the bytes representing `Self`.
    #[inline]
    fn as_byte_ptr(&self) -> *const MaybeUninit<u8> {
        self as *const Self as *const MaybeUninit<u8>
    }

    /// Construct `Self` from a mut pointer to bytes.
    ///
    /// # Safety
    ///
    /// This is inherently unsafe, especially for pointer types since `bytes` can store arbitrary
    /// data. In general, this `fn` is only required to guarantee safety for `bytes` returned by
    /// the `as_byte_mut_ptr` function.
    #[inline]
    unsafe fn from_byte_mut_ptr<'a>(bytes: *mut MaybeUninit<u8>) -> &'a mut Self {
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
    unsafe fn from_byte_ptr<'a>(bytes: *const MaybeUninit<u8>) -> &'a Self {
        &*(bytes as *const Self)
    }

    /// Get a mutable slice of bytes representing `Self`.
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { slice::from_raw_parts_mut(self.as_byte_mut_ptr(), size_of::<Self>()) }
    }

    /// Get a slice of bytes representing `Self`.
    #[inline]
    fn as_bytes(&self) -> &[MaybeUninit<u8>] {
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
    unsafe fn from_bytes_mut(bytes: &mut [MaybeUninit<u8>]) -> &mut Self {
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
    unsafe fn from_bytes(bytes: &[MaybeUninit<u8>]) -> &Self {
        assert_eq!(bytes.len(), size_of::<Self>());
        Self::from_byte_ptr(bytes.as_ptr())
    }

    #[inline]
    fn rc_into_rc_bytes(rc: Rc<Self>) -> Rc<[MaybeUninit<u8>]> {
        let byte_ptr = Rc::into_raw(rc) as *const MaybeUninit<u8>;
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { Rc::from_raw(slice::from_raw_parts(byte_ptr, size_of::<Self>())) }
    }

    #[inline]
    unsafe fn rc_from_rc_bytes(rc: Rc<[MaybeUninit<u8>]>) -> Rc<Self> {
        Rc::from_raw(Rc::into_raw(rc) as *const Self)
    }

    #[inline]
    fn arc_into_arc_bytes(arc: Arc<Self>) -> Arc<[MaybeUninit<u8>]> {
        let byte_ptr = Arc::into_raw(arc) as *const MaybeUninit<u8>;
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { Arc::from_raw(slice::from_raw_parts(byte_ptr, size_of::<Self>())) }
    }

    #[inline]
    unsafe fn arc_from_arc_bytes(arc: Arc<[MaybeUninit<u8>]>) -> Arc<Self> {
        Arc::from_raw(Arc::into_raw(arc) as *const Self)
    }

    #[inline]
    fn box_into_box_bytes(b: Box<Self>) -> Box<[MaybeUninit<u8>]> {
        let byte_ptr = Box::into_raw(b) as *mut MaybeUninit<u8>;
        // This is safe since any memory can be represented by bytes and we are looking at
        // sized types only.
        unsafe { Box::from_raw(slice::from_raw_parts_mut(byte_ptr, size_of::<Self>())) }
    }

    #[inline]
    unsafe fn box_from_box_bytes(b: Box<[MaybeUninit<u8>]>) -> Box<Self> {
        Box::from_raw(Box::into_raw(b) as *mut Self)
    }

    #[inline]
    fn try_into_usize(&self) -> Option<MaybeUninit<usize>> {
        // This is safe since all bit representations with size `size_of::<usize>` are valid usize
        // values.
        unsafe {
            if size_of::<Self>() == size_of::<MaybeUninit<usize>>() {
                Some(std::mem::transmute_copy(self))
            } else {
                None
            }
        }
    }

    /// This function returns `None` if the size of `Self` is not the same as the size of `usize`.
    /// All other checks are left up to the user.
    #[inline]
    unsafe fn try_from_usize(b: MaybeUninit<usize>) -> Option<Self> {
        if size_of::<Self>() == size_of::<MaybeUninit<usize>>() {
            Some(std::mem::transmute_copy(&b))
        } else {
            None
        }
    }
}

// It is assumed that any Rust type has a valid representation in bytes. This library has an
// inherently more relaxed requirement than crates like [`zerocopy`] or [`bytemuck`] since the
// representative bytes cannot be modified or inspected by the safe API exposed by this library,
// they can only be copied.
//
// Further, the bytes representing a type are never interpreted as
// anything other than a type with an identical `TypeId`, which are assumed to have an identical
// memory layout throughout the execution of the program.
//
// For non-Copy types, memory safety is ensured by effectively forgetting reinterpreted memory,
// thus inhibiting any drop calls, which would otherwise could cause dangling references when the
// memory is reinterpreted back into the original type.
//
// In summary, reinterpreting types as bytes is dangerous, but it should not cause undefined
// behavior if it is done carefully.
//
// [`bytemuck`]: https://crates.io/crates/bytemuck
// [`zerocopy`]: https://crates.io/crates/zerocopy
impl<T> Bytes for T where T: 'static {}
