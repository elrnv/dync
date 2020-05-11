//! This crate defines a buffer data structure optimized to be written to and read from standard
//! `Vec`s.
//!
//! [`VecCopy`] is particularly useful when dealing with plain data whose type is determined at
//! run time.  Note that data is stored in the underlying byte buffers in native endian form, thus
//! requesting typed data from a buffer on a platform with different endianness is unsafe.
//!
//! # Caveats
//!
//! [`VecCopy`] doesn't support zero-sized types.
//!
//! [`VecCopy`]: struct.VecCopy

pub mod macros;
mod bytes;
pub mod traits;
#[macro_use]
mod value;
pub mod index_slice;
mod slice_copy;
mod slice_drop;
mod vec_copy;
mod vec_drop;

pub use downcast_rs as downcast;
pub use dync_derive::dync_trait;
pub use index_slice::*;
pub use slice_copy::*;
pub use slice_drop::*;
pub use value::*;
pub use vec_copy::*;
pub use vec_drop::*;

/// Convert a given container with a dynamic vtable to a concrete type.
///
/// This macro will panic if the conversion fails.
#[macro_export]
macro_rules! from_dyn {
    (SliceDrop < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@slice SliceDrop < dyn $trait as $vtable >]
    }};
    (SliceDropMut < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@slice SliceDropMut < dyn $trait as $vtable >]
    }};
    (VecDrop < dyn $trait:path as $vtable:path >) => {{
        from_dyn![@owned VecDrop < dyn $trait as $vtable >]
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
            use std::boxed::Box as Ptr;
            unsafe {
                let (data, size, id, vtable) = vec.into_raw_parts();
                let updated_vtable: Ptr<V> = vtable.downcast().ok().unwrap();
                $vec::from_raw_parts(data, size, id, updated_vtable)
            }
        }

        from_dyn::<$vtable>
    }};
    (@slice $slice:ident < dyn $trait:path >) => {{
        fn from_dyn<'a, V: ?Sized + HasDrop + std::any::Any>(slice: $crate::$slice<'a, V>) -> $crate::$slice<'a, $vtable> {
            unsafe {
                let (data, size, id, vtable) = slice.into_raw_parts();
                match vtable {
                    $crate::VTableRef::Ref(v) => {
                        let updated_vtable: &$vtable = v.downcast_ref().unwrap();
                        $slice::from_raw_parts(data, size, id, updated_vtable)
                    }
                    $crate::VTableRef::Box(v) => {
                        let updated_vtable: Box<$vtable> = v.downcast().unwrap();
                        $slice::from_raw_parts(data, size, id, updated_vtable)
                    }
                    //$crate::VTableRef::Rc(v) => {
                    //    let updated_vtable: std::rc::Rc<$vtable> = v.downcast().unwrap();
                    //    $slice::from_raw_parts(data, size, id, updated_vtable)
                    //}
                }
            }
        }

        from_dyn
    }};
}

/// Convert a given container type (e.g. `VecCopy` or `SliceDyn`) to have a dynamic VTable.
#[macro_export]
macro_rules! into_dyn {
    (SliceDrop < dyn $trait:path >) => {{
        into_dyn![@slice SliceDrop < dyn $trait >]
    }};
    (SliceDropMut < dyn $trait:ident >) => {{
        into_dyn![@slice SliceDropMut < dyn $trait >]
    }};
    (VecDrop < dyn $trait:ident >) => {{
        into_dyn![@owned VecDrop < dyn $trait >]
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
            use std::boxed::Box as Ptr;
            unsafe {
                let (data, size, id, vtable) = vec.into_raw_parts();
                let updated_vtable: Ptr<dyn $trait> = vtable;
                $vec::from_raw_parts(data, size, id, updated_vtable)
            }
        }

        into_dyn
    }};
    (@slice $slice:ident < dyn $trait:path >) => {{
        fn into_dyn<'a, V: 'static + $trait>(slice: $crate::$slice<'a, V>) -> $crate::$slice<'a, dyn $trait> {
            unsafe {
                let (data, size, id, vtable) = slice.into_raw_parts();
                match vtable {
                    $crate::VTableRef::Ref(v) => {
                        let updated_vtable: &dyn $trait = v;
                        $slice::from_raw_parts(data, size, id, updated_vtable)
                    }
                    $crate::VTableRef::Box(v) => {
                        let updated_vtable: Box<dyn $trait> = v;
                        $slice::from_raw_parts(data, size, id, updated_vtable)
                    }
                    //$crate::VTableRef::Rc(v) => {
                    //    let updated_vtable: std::rc::Rc<dyn $trait> = v;
                    //    $slice::from_raw_parts(data, size, id, updated_vtable)
                    //}
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
    fn bytes(&self) -> &[u8];

    /// The size of an element in bytes.
    fn element_size(&self) -> usize;

    /// Get a range of byte indices representing the given element index.
    #[inline]
    fn index_byte_range(&self, i: usize) -> std::ops::Range<usize> {
        i * self.element_size()..(i + 1) * self.element_size()
    }

    /// Index into an immutable slice of bytes.
    #[inline]
    fn index_byte_slice(&self, i: usize) -> &[u8] {
        &self.bytes()[self.index_byte_range(i)]
    }
}

/// A helper trait for mutably accessing internal byte representations of elements represented as
/// contiguous byte slices.
pub(crate) trait ElementBytesMut: ElementBytes {
    /// Get the mutable slice of bytes representing all the elements.
    fn bytes_mut(&mut self) -> &mut [u8];

    /// Index into a mutable slice of bytes.
    #[inline]
    fn index_byte_slice_mut(&mut self, i: usize) -> &mut [u8] {
        let rng = self.index_byte_range(i);
        &mut self.bytes_mut()[rng]
    }

    /// Swap elements at indicies `i` and `j` represented by the bytes.
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        assert_ne!(i, j);
        let element_size = self.element_size();
        let r_rng = self.index_byte_range(0);
        if i < j {
            let l_rng = self.index_byte_range(i);
            let (l, r) = self.bytes_mut().split_at_mut(element_size * j);
            l[l_rng].swap_with_slice(&mut r[r_rng])
        } else {
            let l_rng = self.index_byte_range(j);
            let (l, r) = self.bytes_mut().split_at_mut(element_size * i);
            l[l_rng].swap_with_slice(&mut r[r_rng])
        }
    }
}
