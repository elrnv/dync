//! This module defines a barebones internal API
//! for representing `Vec`s using a void pointer and explicit type infomration about each element.

use std::mem::{align_of, size_of, ManuallyDrop, MaybeUninit};

use crate::elem::*;

/// A decomposition of a `Vec` into a void pointer, a length and a capacity.
///
/// This type exposes a purely internal API since it has no mechanisms for dropping
/// non-copy elements. For a `Vec` type handling copy types, use `VecCopy`.
///
/// This type leaks memory, it is meant to be dropped by a containing type.
#[derive(Debug)]
pub struct VecVoid {
    /// Owned pointer.
    pub(crate) ptr: *mut (),
    /// Vector length.
    pub(crate) len: usize,
    /// Vector capacity.
    pub(crate) cap: usize,
    /// Information about the type of elements stored in this vector.
    pub(crate) elem: ElemInfo,
}

// This implements a shallow clone, which is sufficient for VecCopy, but not for VecDrop.
impl Clone for VecVoid {
    fn clone(&self) -> Self {
        fn clone<T: 'static + Clone>(buf: &VecVoid) -> VecVoid {
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                // Copy all pointers and data from buf. This causes temporary
                // aliasing.  This is safe because in the following we don't
                // modify the original Vec in any way or make any calls to
                // alloc/dealloc. We use mut here solely for the purpose
                // of constructing a `Vec` from a `VoidVec` in order to use
                // `Vec`s clone method.
                let out = VecVoid {
                    ptr: buf.ptr,
                    len: buf.len,
                    cap: buf.cap,
                    elem: buf.elem,
                };
                // Next we clone this new VecVoid. This is basically a memcpy
                // of the internal data into a new heap block.
                let v = ManuallyDrop::new(out.into_aligned_vec_unchecked::<T>());

                // Ensure that the capacity is preserved.
                let mut new_v = Vec::with_capacity(v.capacity());
                new_v.clone_from(&v);

                VecVoid::from_vec_override(new_v, buf.elem)
            }
        }
        eval_align!(self.elem.alignment; clone::<_>(self))
    }

    fn clone_from(&mut self, source: &Self) {
        fn clone_from<T: 'static + Clone>(out: &mut VecVoid, source: &VecVoid) {
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                // Same technique as in `clone`, see comments there.
                let src = VecVoid {
                    ptr: source.ptr,
                    len: source.len,
                    cap: source.cap,
                    elem: source.elem,
                };
                let src = ManuallyDrop::new(src.into_aligned_vec_unchecked::<T>());
                out.apply_aligned_unchecked(|out: &mut Vec<T>| out.clone_from(&src))
            }
        }
        eval_align!(self.elem.alignment; clone_from::<_>(self, source))
    }
}

impl Default for VecVoid {
    fn default() -> Self {
        let v: Vec<()> = Vec::new();
        VecVoid::from_vec(v)
    }
}

impl VecVoid {
    /// Returns the length of this vector.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this vector is empty and false otherwise.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of this vector.
    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    /// Converts the given `Vec<T>` into a `ManuallyDrop` version with the expected allocation
    /// strategy.
    #[inline]
    fn into_valid_vec<T: 'static>(v: Vec<T>) -> ManuallyDrop<Vec<T>> {
        // IMPORTANT:
        // When v has no capacity, then we can't ensure that pushing to it will keep capacity T aligned.
        // So here we push an uninitialized item to it to ensure that capacity will be properly tracked.
        // For this reason converting a Vec to VecVoid causes additional allocations if v is non-empty.
        // Not doing the reserve step below _may_ cause a panic when pushing to a VecVoid.
        let mut v = ManuallyDrop::new(v);
        if v.capacity() == 0 {
            v.reserve(1);
        }
        v
    }

    /// Converts a typed `Vec<T>` into a `VecVoid`.
    ///
    /// If the given `Vec<T>` has no capacity, this function allocates capacity for (at least) a single element
    /// to ensure proper memory alignment in the future.
    #[inline]
    pub(crate) fn from_vec<T: 'static>(v: Vec<T>) -> Self {
        let mut v = Self::into_valid_vec(v);
        VecVoid {
            ptr: v.as_mut_ptr() as *mut (),
            len: v.len(),
            cap: v.capacity(),
            elem: ElemInfo::new::<T>(),
        }
    }

    /// Converts a typed `Vec<T>` into a `VecVoid` while oeverriding the element info of `T` with the
    /// given element info.
    ///
    /// This function expects `v` to have non-zero capacity. Otherwise we cannot ensure that future
    /// allocations will be element aligned with the original type.
    ///
    /// # Safety
    ///
    /// The given element info must be consistently sized and aligned with `T`.
    ///
    /// # Panics
    ///
    /// This function panics if the length of capacity of `v` is not a multiple of the given element size.
    /// It also panics if the capacity of `v` is not large enough to contain one element specified by `elem`.
    #[inline]
    pub(crate) unsafe fn from_vec_override<T: 'static>(v: Vec<T>, elem: ElemInfo) -> Self {
        let mut v = ManuallyDrop::new(v);

        let velem = ElemInfo::new::<T>();
        assert_eq!(velem.alignment, elem.alignment);

        assert!(v.capacity() >= elem.size);

        assert_eq!(v.len() * velem.size % elem.size, 0);

        let len = v.len() * velem.size / elem.size;

        // This check ensures that capacity has been increased at the correct increment, which
        // may not be true if VecVoid starts off with no capacity to begin with.
        assert_eq!(v.capacity() * velem.size % elem.size, 0);

        let cap = v.capacity() * velem.size / elem.size;
        VecVoid {
            ptr: v.as_mut_ptr() as *mut (),
            len,
            cap,
            elem,
        }
    }

    /// Apply a function to this vector as if it was a `Vec<T>` where `T` has
    /// the same alignment as the type this vector was created with.
    #[inline]
    pub(crate) unsafe fn apply_aligned_unchecked<T: 'static, U>(
        &mut self,
        f: impl FnOnce(&mut Vec<T>) -> U,
    ) -> U {
        let orig_elem = self.elem;
        let mut v = std::mem::take(self).into_aligned_vec_unchecked::<T>();
        let out = f(&mut v); // May allocate/deallocate
                             // Returned default VecVoid value can be safely ignored
        let _ = std::mem::replace(self, VecVoid::from_vec_override(v, orig_elem));
        out
    }

    /// Apply a function to this vector as if it was a `Vec<T>` where `T` is
    /// the same as the type this vector was created with.
    #[inline]
    pub(crate) unsafe fn apply_unchecked<T: 'static, U>(
        &mut self,
        f: impl FnOnce(&mut Vec<T>) -> U,
    ) -> U {
        let mut v = std::mem::take(self).into_vec_unchecked::<T>();
        let out = f(&mut v); // May allocate/deallocate
                             // Returned default VecVoid can be safely ignored.
        let _ = std::mem::replace(self, VecVoid::from_vec(v));
        out
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        fn clear<T: 'static>(buf: &mut VecVoid) {
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    v.clear();
                })
            };
        }
        eval_align!(self.elem.alignment; clear::<_>(self));
    }

    /// Reserve `additional` extra elements.
    #[inline]
    pub(crate) fn reserve(&mut self, additional: usize) {
        fn reserve<T: 'static>(buf: &mut VecVoid, additional: usize) {
            let size = buf.elem.size;
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    v.reserve(additional * size);
                })
            };
        }
        eval_align!(self.elem.alignment; reserve::<_>(self, additional));
    }

    #[inline]
    pub(crate) fn truncate(&mut self, new_len: usize) {
        fn truncate<T: 'static>(buf: &mut VecVoid, new_len: usize) {
            let size = buf.elem.size;
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    v.truncate(new_len * size);
                })
            };
        }
        eval_align!(self.elem.alignment; truncate::<_>(self, new_len));
    }

    #[inline]
    pub(crate) fn append(&mut self, other: &mut VecVoid) {
        fn append<T: 'static>(buf: &mut VecVoid, other: &mut VecVoid) {
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|target: &mut Vec<T>| {
                    other.apply_aligned_unchecked(|source: &mut Vec<T>| {
                        target.append(source);
                    });
                });
            }
        }

        // Ensure that the two Vec types are the same.
        assert_eq!(self.elem.type_id, other.elem.type_id);
        debug_assert_eq!(self.elem.alignment, other.elem.alignment);
        debug_assert_eq!(self.elem.size, other.elem.size);

        eval_align!(self.elem.alignment; append::<_>(self, other));
    }

    #[inline]
    pub(crate) fn push(&mut self, val: &[MaybeUninit<u8>]) {
        fn push<T: 'static + Copy>(buf: &mut VecVoid, val: &[MaybeUninit<u8>]) {
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    // SAFETY: T here is one of the T# so it's copy. The
                    // following is effectively a move and doesn't rely on the
                    // "actual" type being Copy.
                    let val_t = *(val.as_ptr() as *const T);
                    v.push(val_t);
                });
            }
        }

        eval_align!(self.elem.alignment; push::<_>(self, val));
    }

    /// Resize this `VecVoid` using a given per element initializer.
    #[cfg(feature = "traits")]
    #[inline]
    pub(crate) unsafe fn resize_with<I>(&mut self, len: usize, init: I)
    where
        I: FnOnce(&mut [MaybeUninit<u8>]),
    {
        fn resize_with<T: 'static + Default + Copy + std::fmt::Debug, Init>(
            buf: &mut VecVoid,
            len: usize,
            init: Init,
        ) where
            Init: FnOnce(&mut [MaybeUninit<u8>]),
        {
            let size = buf.elem.size;
            let num_bytes = buf.elem.num_bytes();
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    // SAFETY: T here is one of the T# so it's copy.
                    let orig_len = v.len();
                    v.resize_with(len * size, Default::default);
                    let byte_slice = std::slice::from_raw_parts_mut(
                        (&mut v[orig_len..]).as_mut_ptr() as *mut MaybeUninit<u8>,
                        num_bytes,
                    );
                    init(byte_slice);
                });
            }
        }

        eval_align!(self.elem.alignment; resize_with::<_, I>(self, len, init));
    }

    #[inline]
    pub(crate) fn rotate_left(&mut self, n: usize) {
        fn rotate_left<T: 'static>(buf: &mut VecVoid, n: usize) {
            let size = buf.elem.size;
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    v.rotate_left(n * size);
                });
            }
        }

        eval_align!(self.elem.alignment; rotate_left::<_>(self, n));
    }

    #[inline]
    pub(crate) fn rotate_right(&mut self, n: usize) {
        fn rotate_right<T: 'static>(buf: &mut VecVoid, n: usize) {
            let size = buf.elem.size;
            // SAFETY: Memory layout is ensured to be correct by eval_align.
            unsafe {
                buf.apply_aligned_unchecked(|v: &mut Vec<T>| {
                    v.rotate_right(n * size);
                });
            }
        }

        eval_align!(self.elem.alignment; rotate_right::<_>(self, n));
    }

    /// Cast this vector into a `Vec<T>` where `T` is alignment sized.
    ///
    /// # Safety
    ///
    /// Trying to interpret the values contained in the underlying `Vec` can
    /// cause undefined behavior, however truncating and reserving space is
    /// valid so long as `T` has the same alignment as `self.elem.alignment`.
    /// This is checked in debug builds.
    pub(crate) unsafe fn into_aligned_vec_unchecked<T>(self) -> Vec<T> {
        let mut md = ManuallyDrop::new(self);
        ManuallyDrop::into_inner(Self::aligned_vec_unchecked(&mut md))
    }

    /// Cast this vector into a `Vec<T>` where `T` is alignment sized.
    ///
    /// # Safety
    ///
    /// Trying to interpret the values contained in the underlying `Vec` can
    /// cause undefined behavior, however truncating and reserving space is
    /// valid so long as `T` has the same alignment as `self.elem.alignment`.
    /// This is checked in debug builds.
    pub(crate) unsafe fn aligned_vec_unchecked<T>(&mut self) -> ManuallyDrop<Vec<T>> {
        debug_assert_eq!(size_of::<T>(), self.elem.alignment);
        debug_assert_eq!(align_of::<T>(), self.elem.alignment);
        let len = self.len * self.elem.size;
        let cap = self.cap * self.elem.size;
        // Make sure the Vec isn't dropped, otherwise it will cause a double
        // free since self is still valid.
        ManuallyDrop::new(Vec::from_raw_parts(self.ptr as *mut T, len, cap))
    }

    /// Cast this vector into a `Vec<T>`.
    ///
    /// # Safety
    ///
    /// Trying to interpret the values contained in the underlying `Vec` can
    /// cause undefined behavior, however truncating and reserving space is
    /// valid so long as `T` has the same size and alignment as given by
    /// `self.elem`.  This is checked in debug builds.
    pub(crate) unsafe fn into_vec_unchecked<T>(self) -> Vec<T> {
        debug_assert_eq!(size_of::<T>(), self.elem.num_bytes());
        debug_assert_eq!(align_of::<T>(), self.elem.alignment);
        let md = ManuallyDrop::new(self);
        Vec::from_raw_parts(md.ptr as *mut T, md.len, md.cap)
    }
}

impl Drop for VecVoid {
    fn drop(&mut self) {
        fn drop_vec<T>(buf: &mut VecVoid) {
            unsafe {
                let _ = ManuallyDrop::into_inner(buf.aligned_vec_unchecked::<T>());
            }
        }
        eval_align!(self.elem.alignment; drop_vec::<_>(self));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_from() {
        let a = VecVoid::from_vec(vec![1u32, 2, 3]);
        let mut b = VecVoid::from_vec(Vec::<u32>::new());
        b.clone_from(&a);
        assert_eq!(a.len, b.len);
        // Capacity may be different.
        assert_eq!(a.elem, b.elem);
        assert_ne!(a.ptr, b.ptr); // Different memory since this is a clone
    }
}
