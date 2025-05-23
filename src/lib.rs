//! # Overview
//!
//! This crate aims to fill the gap in Rust's dynamic traits system by exposing the control over dynamic
//! virtual function tables to the user in a safe API. Below is a list of capabilities unlocked by
//! `dync`.
//!
//! - Create homogeneous untyped `Vec`s that store a single virtual function table for all contained
//!   elements. This functionality is enabled by the `traits` feature. For more details see
//!   [`vec_dyn`].
//!
//! [`vec_dyn`]: vec_dyn/index.html

mod bytes;
pub mod macros;

#[macro_use]
mod elem;
#[macro_use]
mod copy_value;
mod vtable;

mod meta;

#[cfg(feature = "traits")]
pub mod traits;

#[cfg(feature = "traits")]
#[macro_use]
mod value;

pub mod index_slice;
mod vec_void;

mod slice_copy;
mod vec_copy;

#[cfg(feature = "traits")]
mod slice;

#[cfg(feature = "traits")]
mod vec_dyn;

#[cfg(feature = "traits")]
pub use crate::meta::*;
pub use copy_value::*;
#[cfg(feature = "traits")]
pub use downcast_rs as downcast;
#[cfg(feature = "traits")]
pub use dync_derive::dync_mod;
#[cfg(feature = "traits")]
pub use dync_derive::dync_trait;
pub use elem::CopyElem;
pub use index_slice::*;
#[cfg(feature = "traits")]
pub use slice::*;
pub use slice_copy::*;
#[cfg(feature = "traits")]
pub use value::*;
pub use vec_copy::VecCopy;
#[cfg(feature = "traits")]
pub use vec_dyn::*;
pub use vtable::*;

/// Convert a given container with a dynamic vtable to a concrete type.
///
/// This macro will panic if the conversion fails.
#[cfg(feature = "traits")]
#[macro_export]
macro_rules! from_dyn {
    (Slice < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@slice Slice < dyn $trait as $vtable >]
    }};
    (SliceMut < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@slice SliceMut < dyn $trait as $vtable >]
    }};
    (VecDyn < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@owned VecDyn < dyn $trait as $vtable >]
    }};
    (SliceCopy < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@slice SliceCopy < dyn $trait as $vtable >]
    }};
    (SliceCopyMut < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@slice SliceCopyMut < dyn $trait as $vtable >]
    }};
    (VecCopy < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@owned VecCopy < dyn $trait as $vtable >]
    }};
    (@owned $vec:ident < dyn $trait:path as $vtable:path>) => {{
        fn from_dyn<V: $trait>(vec: $crate::$vec<dyn $trait>) -> $crate::$vec<V> {
            unsafe {
                let (data, vtable) = vec.into_raw_parts();
                // If vtables were shared with Rc, we would use this:
                //let updated_vtable: std::rc::Rc<V> = vtable.downcast_rc().ok().unwrap();
                let updated_vtable: Box<V> = vtable.downcast().ok().unwrap();
                $vec::from_raw_parts(data, updated_vtable)
            }
        }

        from_dyn::<$vtable>
    }};
    (@slice $slice:ident < dyn $trait:path >) => {{
        fn from_dyn<'a, V: ?Sized + HasDrop + std::any::Any>(slice: $crate::$slice<'a, V>) -> $crate::$slice<'a, $vtable> {
            unsafe {
                let (data, elem, vtable) = slice.into_raw_parts();
                match vtable {
                    $crate::VTableRef::Ref(v) => {
                        let updated_vtable: &$vtable = v.downcast_ref().unwrap();
                        $slice::from_raw_parts(data, elem, updated_vtable)
                    }
                    $crate::VTableRef::Box(v) => {
                        let updated_vtable: Box<$vtable> = v.downcast().unwrap();
                        $slice::from_raw_parts(data, elem, updated_vtable)
                    }
                    // $crate::VTableRef::Rc(v) => {
                    //     let updated_vtable: std::rc::Rc<$vtable> = v.downcast().unwrap();
                    //     $slice::from_raw_parts(data, elem, updated_vtable)
                    // }
                }
            }
        }

        from_dyn
    }};
}

/// Convert a given container type (e.g. `VecCopy` or `SliceDyn`) to have a dynamic VTable.
#[cfg(feature = "traits")]
#[macro_export]
macro_rules! into_dyn {
    (Slice < dyn $trait:path >) => {{
        into_dyn![@slice Slice < dyn $trait >]
    }};
    (SliceMut < dyn $trait:ident >) => {{
        into_dyn![@slice SliceMut < dyn $trait >]
    }};
    (VecDyn < dyn $trait:ident >) => {{
        into_dyn![@owned VecDyn < dyn $trait >]
    }};
    (SliceCopy < dyn $trait:ident >) => {{
        into_dyn![@slice SliceCopy < dyn $trait >]
    }};
    (SliceCopyMut < dyn $trait:ident >) => {{
        into_dyn![@slice SliceCopyMut < dyn $trait >]
    }};
    (VecCopy < dyn $trait:ident >) => {{
        into_dyn![@owned VecCopy < dyn $trait >]
    }};
    (@owned $vec:ident < dyn $trait:ident >) => {{
        fn into_dyn<V: 'static + $trait>(vec: $crate::$vec<V>) -> $crate::$vec<dyn $trait> {
            unsafe {
                let (data, vtable) = vec.into_raw_parts();
                // If vtables were shared with Rc, we would use this:
                //let updated_vtable: std::rc::Rc<dyn $trait> = vtable;
                let updated_vtable: Box<dyn $trait> = vtable;
                $vec::from_raw_parts(data, updated_vtable)
            }
        }

        into_dyn
    }};
    (@slice $slice:ident < dyn $trait:path >) => {{
        fn into_dyn<'a, V: 'static + $trait>(slice: $crate::$slice<'a, V>) -> $crate::$slice<'a, dyn $trait> {
            unsafe {
                let (data, elem, vtable) = slice.into_raw_parts();
                match vtable {
                    $crate::VTableRef::Ref(v) => {
                        let updated_vtable: &dyn $trait = v;
                        $slice::from_raw_parts(data, elem, updated_vtable)
                    }
                    $crate::VTableRef::Box(v) => {
                        let updated_vtable: Box<dyn $trait> = v;
                        $slice::from_raw_parts(data, elem, updated_vtable)
                    }
                    // $crate::VTableRef::Rc(v) => {
                    //     let updated_vtable: std::rc::Rc<dyn $trait> = v;
                    //     $slice::from_raw_parts(data, elem, updated_vtable)
                    // }
                }
            }
        }

        into_dyn
    }};
}

/// A helper trait for accessing internal byte representations of elements represented as
/// contiguous byte slices.
pub(crate) trait ElementBytes {
    /// Get the slice of bytes representing all the elements.
    fn bytes(&self) -> &[std::mem::MaybeUninit<u8>];

    /// The size of an element in bytes.
    fn element_size(&self) -> usize;

    /// Get a range of byte indices representing the given element index.
    #[inline]
    fn index_byte_range(&self, i: usize) -> std::ops::Range<usize> {
        i * self.element_size()..(i + 1) * self.element_size()
    }

    /// Index into an immutable slice of bytes.
    #[inline]
    fn index_byte_slice(&self, i: usize) -> &[std::mem::MaybeUninit<u8>] {
        &self.bytes()[self.index_byte_range(i)]
    }
}

/// A helper trait for mutably accessing internal byte representations of elements represented as
/// contiguous byte slices.
pub(crate) trait ElementBytesMut: ElementBytes {
    /// Get the mutable slice of bytes representing all the elements.
    unsafe fn bytes_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>];

    /// Index into a mutable slice of bytes.
    #[inline]
    unsafe fn index_byte_slice_mut(&mut self, i: usize) -> &mut [std::mem::MaybeUninit<u8>] {
        let rng = self.index_byte_range(i);
        &mut self.bytes_mut()[rng]
    }

    /// Swap elements at indicies `i` and `j` represented by the bytes.
    ///
    /// If `i` is the same as `j` this function does nothing.
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        let element_size = self.element_size();
        let r_rng = self.index_byte_range(0);
        if i < j {
            let l_rng = self.index_byte_range(i);
            // SAFETY: it is safe to swap aligned data since we have unique access.
            unsafe {
                let (l, r) = self.bytes_mut().split_at_mut(element_size * j);
                l[l_rng].swap_with_slice(&mut r[r_rng])
            }
        } else {
            let l_rng = self.index_byte_range(j);
            unsafe {
                let (l, r) = self.bytes_mut().split_at_mut(element_size * i);
                l[l_rng].swap_with_slice(&mut r[r_rng])
            }
        }
    }
}
