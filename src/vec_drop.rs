//! This module extends the `VecCopy` type to more general non-`Copy` types that can be `Drop`ped.
//!
//! This module is enabled by the `traits` feature.
//!
//! # Examples
//!
//! Create homogeneous untyped `Vec`s that store a single virtual function table for all contained
//! elements:
//! ```
//! use dync::VecDrop;
//! // Create an untyped `Vec`.
//! let vec: VecDrop = vec![1_i32,2,3,4].into();
//! // Access elements either by downcasting to the underlying type.
//! for value_ref in vec.iter() {
//!     let int = value_ref.downcast::<i32>().unwrap();
//!     println!("{}", int);
//! }
//! // Or downcast the iterator directly for more efficient traversal.
//! for int in vec.iter_as::<i32>().unwrap() {
//!     println!("{}", int);
//! }
//! ```
//!
//! The `VecDrop` type defaults to the empty virtual table (with the exception of the drop
//! function), which is not terribly useful when the contained values need to be processed in
//! some way.  `dync` provides support for common standard library traits such as:
//! - `Drop`
//! - `Clone`
//! - `PartialEq`
//! - `std::hash::Hash`
//! - `std::fmt::Debug`
//! - `Send` and `Sync`
//! - more to come
//!
//! So to produce a `VecDrop` of a printable type, we could instead do
//! ```
//! use dync::{VecDrop, traits::DebugVTable};
//! // Create an untyped `Vec` of `std::fmt::Debug` types.
//! let vec: VecDrop<DebugVTable> = vec![1_i32,2,3,4].into();
//! // We can now iterate and print value references (which inherit the VTable from the container)
//! // without needing a downcast.
//! for value_ref in vec.iter() {
//!     println!("{:?}", value_ref);
//! }
//! ```

#![allow(dead_code)]

use std::{
    any::{Any, TypeId},
    fmt,
    mem::{ManuallyDrop, MaybeUninit},
    slice,
};

// At the time of this writing, there is no evidence that there is a significant benefit in sharing
// vtables via Rc or Arc, but to make potential future refactoring easier we use the Ptr alias.
use std::boxed::Box as Ptr;

#[cfg(feature = "numeric")]
use num_traits::{cast, NumCast, Zero};

use crate::meta::*;
use crate::slice_drop::*;
use crate::traits::*;
use crate::value::*;
use crate::vtable::*;
use crate::ElementBytes;
use crate::VecCopy;

pub trait Elem: Any + DropBytes {}
impl<T> Elem for T where T: Any + DropBytes {}

pub struct VecDrop<V = (DropFn, ())>
where
    V: ?Sized + HasDrop,
{
    data: ManuallyDrop<VecCopy<V>>,
}

impl<V: ?Sized + HasDrop> Drop for VecDrop<V> {
    fn drop(&mut self) {
        unsafe {
            {
                // Drop the contents using the associated drop function
                let VecCopy {
                    data,
                    vtable,
                    element_size,
                    ..
                } = &mut *self.data;
                let chunks_iter = data.chunks_exact_mut(*element_size);
                for elem_bytes in chunks_iter {
                    vtable.drop_fn()(elem_bytes);
                }
            }

            // Drop the vec itself
            ManuallyDrop::drop(&mut self.data);
        }
    }
}

impl<V: ?Sized + Clone + HasDrop + HasClone> Clone for VecDrop<V> {
    fn clone(&self) -> Self {
        let data_clone = |bytes: &[MaybeUninit<u8>]| {
            let mut new_data = bytes.to_vec();
            self.data
                .byte_chunks()
                .zip(new_data.chunks_exact_mut(self.data.element_size()))
                .for_each(|(src, dst)| unsafe {
                    // This is safe since `clone_into_raw_fn` ensures that the
                    // bytes in dst are not dropped before cloning, which is essential, since they
                    // are just copied by the `.to_vec()` call above.
                    self.data.vtable.clone_into_raw_fn()(src, dst)
                });
            new_data
        };
        VecDrop {
            data: ManuallyDrop::new(self.data.clone_with(data_clone)),
        }
    }
}

impl<V: ?Sized + HasDrop + HasPartialEq> PartialEq for VecDrop<V> {
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(this, that)| this == that)
    }
}

impl<V: ?Sized + HasDrop + HasEq> Eq for VecDrop<V> {}

impl<V: ?Sized + HasDrop + HasHash> std::hash::Hash for VecDrop<V> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.iter().for_each(|elem| elem.hash(state));
    }
}

impl<V: ?Sized + HasDrop + HasDebug> fmt::Debug for VecDrop<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

unsafe impl<V: ?Sized + HasDrop + HasSend> Send for VecDrop<V> {}
unsafe impl<V: ?Sized + HasDrop + HasSync> Sync for VecDrop<V> {}

impl<V: HasDrop> VecDrop<V> {
    /// Construct an empty vector with a specific pointed-to element type.
    #[inline]
    pub fn with_type<T: Elem>() -> Self
    where
        V: VTable<T>,
    {
        VecDrop {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::with_type_non_copy::<T>() }),
        }
    }

    /// Construct an empty vector with a capacity for a given number of typed pointed-to elements.
    #[inline]
    pub fn with_capacity<T: Elem>(n: usize) -> Self
    where
        V: VTable<T>,
    {
        VecDrop {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::with_capacity_non_copy::<T>(n) }),
        }
    }

    /// Construct a `VecDrop` from a given `Vec` reusing the space already allocated by the given
    /// vector.
    pub fn from_vec<T: Elem>(vec: Vec<T>) -> Self
    where
        V: VTable<T>,
    {
        VecDrop {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_vec_non_copy(vec) }),
        }
    }
}

impl<V: ?Sized + HasDrop> VecDrop<V> {
    /// Construct a vector with the same type as the given vector without copying its data.
    #[inline]
    pub fn with_type_from(other: impl Into<Meta<Ptr<V>>>) -> Self {
        VecDrop {
            data: ManuallyDrop::new(VecCopy::with_type_from(other.into())),
        }
    }

    /// Construct a `VecDrop` from raw bytes and type metadata.
    ///
    /// # Safety
    ///
    /// Almost exclusively the only inputs that are safe here are the ones returned by
    /// `VecDrop::into_raw_parts`.
    ///
    /// This function should not be used other than in internal APIs. It exists to enable the
    /// `into_dyn` macro until `CoerceUsize` is stabilized.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: Vec<MaybeUninit<u8>>,
        element_size: usize,
        element_type_id: TypeId,
        vtable: Ptr<V>,
    ) -> VecDrop<V> {
        VecDrop {
            data: ManuallyDrop::new(VecCopy {
                data,
                element_size,
                element_type_id,
                vtable,
            }),
        }
    }

    /// Convert this collection into its raw components.
    ///
    /// This function exists mainly to enable the `into_dyn` macro until `CoerceUnsized` is
    /// stabilized.
    #[inline]
    pub fn into_raw_parts(self) -> (Vec<MaybeUninit<u8>>, usize, TypeId, Ptr<V>) {
        unsafe {
            // Inhibit dropping self.
            let mut md = ManuallyDrop::new(self);
            // Taking is safe here because data will not be used after this call since self is
            // consumed, and self will not be dropped.
            let VecCopy {
                data,
                element_size,
                element_type_id,
                vtable,
            } = ManuallyDrop::take(&mut md.data);
            (data, element_size, element_type_id, vtable)
        }
    }

    /// Retrieve the associated virtual function table.
    pub fn vtable(&self) -> &V {
        &self.data.vtable
    }
    /// Upcast the `VecDrop` into a more general base `VecDrop`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: HasDrop + From<V>>(self) -> VecDrop<U>
    where
        V: Clone,
    {
        // Inhibit dropping. The output vec takes ownership.
        let mut md = ManuallyDrop::new(self);
        // This is safe since self will not be dropped, so md.data will not be dropped.
        let data = unsafe { ManuallyDrop::take(&mut md.data) };
        VecDrop {
            data: ManuallyDrop::new(data.upcast()), //_with(|(drop, v)| (drop, U::from(v)))),
        }
    }

    /// Clear the data buffer without destroying its type information.
    #[inline]
    pub fn clear(&mut self) {
        // Drop all elements manually.
        unsafe {
            let VecCopy {
                data,
                vtable,
                element_size,
                ..
            } = &mut *self.data;
            let chunks_iter = data.chunks_exact_mut(*element_size);
            for elem_bytes in chunks_iter {
                vtable.drop_fn()(elem_bytes);
            }
            data.clear();
        }
    }

    /// Add an element to this buffer.
    ///
    /// If the type of the given element coincides with the type stored by this buffer,
    /// then the modified buffer is returned via a mutable reference.  Otherwise, `None` is
    /// returned.
    #[inline]
    pub fn push_as<T: Elem>(&mut self, element: T) -> Option<&mut Self> {
        if self.data.push_as(element).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Check if the current buffer contains elements of the specified type. Returns `Some(self)`
    /// if the type matches and `None` otherwise.
    #[inline]
    pub fn check<T: Elem>(self) -> Option<Self> {
        if self.data.check_ref::<T>().is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Check if the current buffer contains elements of the specified type. Returns `None` if the
    /// check fails, otherwise a reference to self is returned.
    #[inline]
    pub fn check_ref<T: Elem>(&self) -> Option<&Self> {
        self.data.check_ref::<T>().map(|_| self)
    }

    /// Check if the current buffer contains elements of the specified type. Same as `check_ref`
    /// but consumes and produces a mut reference to self.
    #[inline]
    pub fn check_mut<T: Elem>(&mut self) -> Option<&mut Self> {
        if self.data.check_mut::<T>().is_some() {
            Some(self)
        } else {
            None
        }
    }

    /*
     * Accessors
     */

    /// Get the `TypeId` of data stored within this buffer.
    #[inline]
    pub fn element_type_id(&self) -> TypeId {
        self.data.element_type_id()
    }

    /// Get the number of elements stored in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if there are any elements stored in this buffer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the byte capacity of this buffer.
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.data.byte_capacity()
    }

    /// Get the size of the element type in bytes.
    #[inline]
    pub fn element_size(&self) -> usize {
        self.data.element_size()
    }

    /// Return an iterator to a slice representing typed data.
    ///
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn iter_as<T: Elem>(&self) -> Option<slice::Iter<T>> {
        self.data.iter_as::<T>()
    }

    /// Return an iterator to a mutable slice representing typed data.
    ///
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn iter_mut_as<T: Elem>(&mut self) -> Option<slice::IterMut<T>> {
        self.data.iter_mut_as::<T>()
    }

    /// An alternative to using the `Into` trait.
    ///
    /// This function helps the compiler determine the type `T` automatically.
    #[inline]
    pub fn into_vec<T: Elem>(self) -> Option<Vec<T>> {
        // This is safe because self.data will not be used after this call, and the resulting
        // Vec<T> will drop all elements correctly.
        unsafe {
            // Inhibit the Drop for self.
            let mut no_drop = ManuallyDrop::new(self);
            // Extract the value from data and turn it into a `Vec` which will handle the drop
            // correctly.
            ManuallyDrop::take(&mut no_drop.data).into_vec()
        }
    }

    /// Convert this buffer into a typed slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice_as<T: Elem>(&self) -> Option<&[T]> {
        self.data.as_slice_as()
    }

    /// Convert this buffer into a typed mutable slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_mut_slice_as<T: Elem>(&mut self) -> Option<&mut [T]> {
        self.data.as_mut_slice_as()
    }

    /// Get a `const` reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_ref_as<T: Elem>(&self, i: usize) -> Option<&T> {
        self.data.get_ref_as::<T>(i)
    }

    /// Get a mutable reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_mut_as<T: Elem>(&mut self, i: usize) -> Option<&mut T> {
        self.data.get_mut_as::<T>(i)
    }

    /// Move bytes to this buffer.
    ///
    /// The given buffer must have the same underlying type as `self`.
    #[inline]
    pub fn append(&mut self, buf: &mut VecDrop<V>) -> Option<&mut Self> {
        // It is sufficient to move the bytes, no clones or drops are necessary here.
        if self.data.append(&mut buf.data).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Rotates the slice in-place such that the first `mid` elements of the slice move to the end
    /// while the last `self.len() - mid` elements move to the front.
    ///
    /// After calling `rotate_left`, the element previously at index `mid` will become the
    /// first element in the slice.
    #[inline]
    pub fn rotate_left(&mut self, mid: usize) {
        self.data.rotate_left(mid)
    }

    /// Rotates the slice in-place such that the first `self.len() - k` elements of the slice move
    /// to the end while the last `k` elements move to the front.
    ///
    /// After calling `rotate_right`, the element previously at index `k` will become the
    /// first element in the slice.
    #[inline]
    pub fn rotate_right(&mut self, k: usize) {
        self.data.rotate_right(k)
    }

    /*
     * Value API. This allows users to manipulate contained data without knowing the element type.
     */

    /// Push a value onto this buffer.
    ///
    /// If the type of the given value coincides with the type stored by this buffer,
    /// then the modified buffer is returned via a mutable reference.  Otherwise, `None` is
    /// returned.
    ///
    /// Note that the vtables need not patch, only the underlying types are required to match.
    #[inline]
    pub fn push<U: ?Sized + HasDrop>(&mut self, value: BoxValue<U>) -> Option<&mut Self> {
        if self.element_type_id() == value.value_type_id() {
            // Prevent the value from being dropped at the end of this scope since it will be later
            // dropped by this container. The remaining fields like vtable will be dropped here.
            let (bytes, _, _) = value.into_raw_parts();
            self.data.data.extend_from_slice(&*bytes);
            Some(self)
        } else {
            None
        }
    }

    /// Push a clone of the referenced value to this buffer.
    ///
    /// If the type of the given value coincides with the type stored by this buffer,
    /// then the modified buffer is returned via a mutable reference.  Otherwise, `None` is
    /// returned.
    ///
    /// This is more efficient than `push_value` since it avoids an extra allocation, however it
    /// requires the contained value to be `Clone`.
    #[inline]
    pub fn push_cloned(&mut self, value: ValueRef<V>) -> Option<&mut Self>
    where
        V: HasClone,
    {
        if self.element_type_id() == value.value_type_id() {
            let orig_len = self.data.data.len();
            self.data
                .data
                .resize(orig_len + value.bytes.len(), MaybeUninit::<u8>::uninit());
            // This does not leak because the copied bytes are guaranteed to be dropped by self.
            // This will also not cause a double free since the bytes in self are not dropped by
            // clone_into_raw_fn unlike clone_from_fn.
            unsafe {
                self.data.vtable.clone_into_raw_fn()(value.bytes, &mut self.data.data[orig_len..]);
            }
            Some(self)
        } else {
            None
        }
    }

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> ValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueRef::from_raw_parts(
                self.data.get_bytes(i),
                self.element_type_id(),
                self.data.vtable.as_ref(),
            )
        }
    }

    /// Return an iterator over untyped value references stored in this buffer.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is
    /// needed for each element.
    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = ValueRef<'a, V>> + 'a {
        let &Self { ref data, .. } = self;
        let VecCopy {
            data,
            element_size,
            element_type_id,
            vtable,
        } = &**data;
        data.chunks_exact(*element_size).map(move |bytes| unsafe {
            ValueRef::from_raw_parts(bytes, *element_type_id, vtable.as_ref())
        })
    }

    /// Get a mutable reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> ValueMut<V> {
        debug_assert!(i < self.len());
        // Safety is guaranteed here by the value API.
        let Self { data, .. } = self;
        let element_bytes = data.index_byte_range(i);
        let &mut VecCopy {
            ref mut data,
            element_type_id,
            ref vtable,
            ..
        } = &mut **data;
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueMut::from_raw_parts(&mut data[element_bytes], element_type_id, vtable.as_ref())
        }
    }

    /// Return an iterator over mutable untyped value references stored in this buffer.
    ///
    /// In contrast to `iter_mut`, this function defers downcasting on a per element basis.  As a
    /// result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    #[inline]
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = ValueMut<'a, V>> + 'a
    where
        V: Clone,
    {
        let VecCopy {
            ref mut data,
            element_size,
            element_type_id,
            ref vtable,
        } = *self.data;
        let vtable = vtable.as_ref();
        data.chunks_exact_mut(element_size)
            .map(move |bytes| unsafe { ValueMut::from_raw_parts(bytes, element_type_id, vtable) })
    }

    pub fn as_slice(&self) -> SliceDrop<V> {
        let VecCopy {
            ref data,
            element_size,
            element_type_id,
            ref vtable,
        } = *self.data;
        unsafe { SliceDrop::from_raw_parts(data, element_size, element_type_id, vtable.as_ref()) }
    }

    pub fn as_mut_slice(&mut self) -> SliceDropMut<V> {
        let VecCopy {
            ref mut data,
            element_size,
            element_type_id,
            ref vtable,
        } = *self.data;
        unsafe {
            SliceDropMut::from_raw_parts(data, element_size, element_type_id, vtable.as_ref())
        }
    }

    /*
     * Advanced Accessors
     */

    /// Get a `const` reference to the `i`'th element of the vector.
    ///
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current vector. See the implementation for details.
    ///
    /// # Safety
    ///
    /// It is assumed that that the vector contains elements of type `T` and that `i` is strictly
    /// less than the length of this vector, otherwise this function may cause undefined behavior.
    ///
    /// This function is a complete opt-out of all safety checks.
    #[inline]
    pub unsafe fn get_unchecked_ref<T: Any>(&self, i: usize) -> &T {
        let ptr = self.data.data.as_ptr() as *const T;
        &*ptr.add(i)
    }

    /// Get a mutable reference to the `i`'th element of the vector.
    ///
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current vector. See the implementation for details.
    ///
    /// # Safety
    ///
    /// It is assumed that that the vector contains elements of type `T` and that `i` is strictly
    /// less than the length of this vector, otherwise this function may cause undefined behavior.
    ///
    /// This function is opts-out of all safety checks.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: Any>(&mut self, i: usize) -> &mut T {
        let ptr = self.data.data.as_mut_ptr() as *mut T;
        &mut *ptr.add(i)
    }
}

// Additional functionality of VecDrops that implement Clone.
impl<V: HasDrop + HasClone> VecDrop<V> {
    /// Construct a typed `DataBuffer` with a given size and filled with the specified default
    /// value.
    #[inline]
    pub fn with_size<T: Elem + Clone>(n: usize, def: T) -> Self
    where
        V: VTable<T>,
    {
        VecDrop {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_vec_non_copy(vec![def; n]) }),
        }
    }

    /// Construct a buffer from a given slice by cloning the data.
    #[inline]
    pub fn from_slice<T: Elem + Clone>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        VecDrop {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_slice_non_copy::<T>(slice) }),
        }
    }
}

impl<V: HasDrop> VecDrop<V> {
    #[cfg(feature = "numeric")]
    /// Cast a numeric `VecDrop` into the given output `Vec` type.
    ///
    /// This only works if the contained element is `Copy`.
    pub fn cast_into_vec<T>(&self) -> Option<Vec<T>>
    where
        T: Elem + Copy + NumCast + Zero,
    {
        use crate::CopyElem;
        // Helper function (generic on the input) to convert the given VecDrop into Vec.
        unsafe fn convert_into_vec<I, O, V>(buf: &VecCopy<V>) -> Option<Vec<O>>
        where
            I: CopyElem + Any + NumCast,
            O: CopyElem + NumCast + Zero,
        {
            debug_assert_eq!(buf.element_type_id(), TypeId::of::<I>()); // Check invariant.
            Some(
                buf.reinterpret_as_slice()
                    .iter()
                    .map(|elem: &I| cast(*elem).unwrap_or_else(O::zero))
                    .collect(),
            )
        }
        call_numeric_buffer_fn!( convert_into_vec::<_, T, V>(&self.data) or { None } )
    }
}

impl<V: ?Sized + HasDrop + HasClone> VecDrop<V> {
    /// Resizes the buffer in-place to store `new_len` elements and returns an optional
    /// mutable reference to `Self`.
    ///
    /// If `value` does not correspond to the underlying element type, then `None` is returned and the
    /// buffer is left unchanged.
    ///
    /// This function has the similar properties to `Vec::resize`.
    #[inline]
    pub fn resize<T: Elem + Clone>(&mut self, new_len: usize, value: T) -> Option<&mut Self> {
        self.check_ref::<T>()?;
        let size_t = std::mem::size_of::<T>();

        if new_len >= self.len() {
            let diff = new_len - self.len();
            self.data.reserve_bytes(diff * size_t);
            for _ in 0..diff {
                self.data.push_as(value.clone());
            }
        } else {
            // Drop trailing elements manually.
            unsafe {
                let VecCopy {
                    data,
                    vtable,
                    element_size,
                    ..
                } = &mut *self.data;
                let chunks_iter = data.chunks_exact_mut(*element_size);
                for elem_bytes in chunks_iter.skip(new_len) {
                    vtable.drop_fn()(elem_bytes);
                }
            }

            // Truncate data
            self.data
                .data
                .resize(new_len * size_t, MaybeUninit::<u8>::uninit());
        }
        Some(self)
    }

    /// Fill the current buffer with clones of the given value.
    ///
    /// The size of the buffer is left unchanged. If the given type doesn't match the
    /// internal type, `None` is returned, otherwise a mutable reference to the modified buffer is
    /// returned.
    #[inline]
    pub fn fill<T: Elem + Clone>(&mut self, def: T) -> Option<&mut Self> {
        for v in self.iter_mut_as::<T>()? {
            *v = def.clone();
        }
        Some(self)
    }

    /// Append cloned items from this buffer to a given `Vec`.
    ///
    /// Return the mutable reference `Some(vec)` if type matched the internal type and
    /// `None` otherwise.
    #[inline]
    pub fn append_cloned_to_vec<'a, T: Elem + Clone>(
        &self,
        vec: &'a mut Vec<T>,
    ) -> Option<&'a mut Vec<T>> {
        let slice = self.as_slice_as()?;
        // Only allocate once we have confirmed that the given `T` matches to avoid unnecessary
        // overhead.
        vec.reserve(self.len());
        vec.extend_from_slice(slice);
        Some(vec)
    }

    /// Clones contents of `self` into the given `Vec`.
    #[inline]
    pub fn clone_into_vec<T: Elem + Clone>(&self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        match self.append_cloned_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }
}

/// Convert a `Vec` to a buffer.
impl<T: Elem, V: HasDrop + VTable<T>> From<Vec<T>> for VecDrop<V> {
    #[inline]
    fn from(vec: Vec<T>) -> VecDrop<V> {
        VecDrop::from_vec(vec)
    }
}

/// Convert a slice to a `VecDrop`.
impl<'a, T, V> From<&'a [T]> for VecDrop<V>
where
    T: Elem + Clone,
    V: HasDrop + VTable<T> + HasClone,
{
    #[inline]
    fn from(slice: &'a [T]) -> VecDrop<V> {
        VecDrop::from_slice(slice)
    }
}

/// Convert a buffer to a `Vec` with an option to fail.
impl<T: Elem, V: ?Sized + HasDrop + VTable<T>> Into<Option<Vec<T>>> for VecDrop<V> {
    #[inline]
    fn into(self) -> Option<Vec<T>> {
        self.into_vec()
    }
}

impl<'a, V: Clone + HasDrop> From<&'a VecDrop<V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(v: &'a VecDrop<V>) -> Self {
        Meta::from(&*v.data)
    }
}

impl<'a, V: Clone + HasDrop> From<&'a VecDrop<V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(v: &'a VecDrop<V>) -> Self {
        Meta::from(&*v.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dync_derive::dync_trait;
    use rand::prelude::*;
    use std::mem::size_of;
    use std::rc::Rc;

    #[dync_trait(dync_crate_name = "crate")]
    pub trait AllTrait: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {}
    impl<T> AllTrait for T where T: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {}
    type VecCopyAll = VecCopy<AllTraitVTable>;
    type VecDropAll = VecDrop<AllTraitVTable>;

    #[inline]
    fn compute(x: i64, y: i64, z: i64) -> [i64; 3] {
        [x - 2 * y + z * 2, y - 2 * z + x * 2, z - 2 * x + y * 2]
    }

    #[inline]
    fn make_random_vec_copy(n: usize) -> VecCopyAll {
        make_random_vec(n).into()
    }

    #[inline]
    fn make_random_vec_dyn(n: usize) -> VecDropAll {
        make_random_vec(n).into()
    }

    #[inline]
    fn make_random_vec(n: usize) -> Vec<[i64; 3]> {
        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let between = rand::distributions::Uniform::from(1i64..5);
        (0..n).map(move |_| [between.sample(&mut rng); 3]).collect()
    }

    #[inline]
    fn vec_copy_compute<V>(v: &mut VecCopy<V>) {
        for a in v.iter_mut() {
            let a = a.downcast::<[i64; 3]>().unwrap();
            let res = compute(a[0], a[1], a[2]);
            a[0] = res[0];
            a[1] = res[1];
            a[2] = res[2];
        }
    }

    #[inline]
    fn vec_dyn_compute<V: Clone + HasDrop>(v: &mut VecDrop<V>) {
        for a in v.iter_mut() {
            let a = a.downcast::<[i64; 3]>().unwrap();
            let res = compute(a[0], a[1], a[2]);
            a[0] = res[0];
            a[1] = res[1];
            a[2] = res[2];
        }
    }

    #[inline]
    fn vec_compute(v: &mut Vec<[i64; 3]>) {
        for a in v.iter_mut() {
            let res = compute(a[0], a[1], a[2]);
            a[0] = res[0];
            a[1] = res[1];
            a[2] = res[2];
        }
    }

    #[test]
    fn downcast_value_mut() {
        use std::time::Instant;
        let size = 90_000;
        let mut v: VecDropAll = make_random_vec_dyn(size);
        let start = Instant::now();
        vec_dyn_compute(&mut v);
        eprintln!("vec_dyn: {} millis", start.elapsed().as_millis());
        let mut v: VecCopyAll = make_random_vec_copy(size);
        let start = Instant::now();
        vec_copy_compute(&mut v);
        eprintln!("vec_copy: {} millis", start.elapsed().as_millis());
        let mut v: Vec<[i64; 3]> = make_random_vec(size);
        let start = Instant::now();
        vec_compute(&mut v);
        eprintln!("vec: {} millis", start.elapsed().as_millis());
    }

    #[test]
    fn clone_from_test() {
        use std::collections::HashSet;

        // Let's create a collection of `Rc`s.
        let vec_rc: Vec<_> = vec![1, 23, 2, 42, 23, 1, 13534653]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec_rc.clone()); // Clone into VecDrop

        // Construct a hashset of unique values from the VecDrop.
        let mut hashset: HashSet<BoxValue<AllTraitVTable>> = HashSet::new();

        for rc_ref in buf.iter().take(4) {
            assert!(hashset.insert(rc_ref.clone_value()));
        }

        assert!(!hashset.insert(BoxValue::new(Rc::clone(&vec_rc[4]))));
        assert!(!hashset.insert(BoxValue::new(Rc::clone(&vec_rc[5]))));

        assert_eq!(hashset.len(), 4);
        assert!(hashset.contains(&BoxValue::new(Rc::new(1))));
        assert!(hashset.contains(&BoxValue::new(Rc::new(23))));
        assert!(hashset.contains(&BoxValue::new(Rc::new(2))));
        assert!(hashset.contains(&BoxValue::new(Rc::new(42))));
        assert!(!hashset.contains(&BoxValue::new(Rc::new(13534653))));
    }

    #[test]
    fn clone_from_small_test() {
        use std::collections::HashSet;

        // Let's create a collection of `Rc`s.
        let vec_rc: Vec<_> = vec![1, 23, 2, 42, 23, 1, 13534653]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec_rc.clone()); // Clone into VecDrop

        // Construct a hashset of unique values from the VecDrop.
        let mut hashset: HashSet<SmallValue<AllTraitVTable>> = HashSet::new();

        for rc_ref in buf.iter().take(4) {
            assert!(hashset.insert(rc_ref.clone_small_value()));
        }

        assert!(!hashset.insert(SmallValue::new(Rc::clone(&vec_rc[4]))));
        assert!(!hashset.insert(SmallValue::new(Rc::clone(&vec_rc[5]))));

        assert_eq!(hashset.len(), 4);
        assert!(hashset.contains(&SmallValue::new(Rc::new(1))));
        assert!(hashset.contains(&SmallValue::new(Rc::new(23))));
        assert!(hashset.contains(&SmallValue::new(Rc::new(2))));
        assert!(hashset.contains(&SmallValue::new(Rc::new(42))));
        assert!(!hashset.contains(&SmallValue::new(Rc::new(13534653))));
    }

    #[test]
    fn iter() {
        let vec: Vec<_> = vec![1, 23, 2, 42, 11].into_iter().map(Rc::new).collect();
        {
            let buf = VecDropAll::from(vec.clone()); // Convert into buffer
            let orig = Rc::new(100);
            let mut rc = Rc::clone(&orig);
            assert_eq!(Rc::strong_count(&rc), 2);
            for val in buf.iter() {
                ValueMut::new(&mut rc).clone_from_other(val).unwrap();
            }
            assert_eq!(Rc::strong_count(&orig), 1);
            assert_eq!(Rc::strong_count(&rc), 3);
            assert_eq!(Rc::strong_count(&vec[4]), 3);
            assert!(vec.iter().take(4).all(|x| Rc::strong_count(x) == 2));
            assert_eq!(rc, Rc::new(11));
        }
        assert!(vec.iter().all(|x| Rc::strong_count(x) == 1));
    }

    /// Test various ways to create a `VecDrop`.
    #[test]
    fn initialization_test() {
        // Empty typed buffer.
        let a = VecDropAll::with_type::<Rc<u8>>();
        assert_eq!(a.len(), 0);
        assert_eq!(a.element_type_id(), TypeId::of::<Rc<u8>>());
        assert_eq!(a.byte_capacity(), 0); // Ensure nothing is allocated.

        // Empty buffer typed by the given type id.
        let b = VecDropAll::with_type_from(&a);
        assert_eq!(b.len(), 0);
        assert_eq!(b.element_type_id(), TypeId::of::<Rc<u8>>());
        assert_eq!(a.byte_capacity(), 0); // Ensure nothing is allocated.

        // Empty typed buffer with a given capacity.
        let a = VecDropAll::with_capacity::<Rc<u8>>(4);
        assert_eq!(a.len(), 0);
        assert_eq!(a.byte_capacity(), 4 * size_of::<Rc<u8>>());
        assert_eq!(a.element_type_id(), TypeId::of::<Rc<u8>>());
    }

    /// Test resizing a buffer.
    #[test]
    fn resize() {
        let mut a = VecDropAll::with_type::<Rc<u8>>();

        // Increase the size of a.
        a.resize(3, Rc::new(1u8))
            .expect("Failed to resize VecDrop up by 3 elements");

        assert_eq!(a.len(), 3);
        for i in 0..3 {
            assert_eq!(a.get_ref_as::<Rc<u8>>(i).unwrap(), &Rc::new(1));
        }

        // Truncate a.
        a.resize(2, Rc::new(1u8))
            .expect("Failed to resize VecDrop down to 2 elements");

        assert_eq!(a.len(), 2);
        for i in 0..2 {
            assert_eq!(a.get_ref_as::<Rc<u8>>(i).unwrap(), &Rc::new(1));
        }
    }

    #[test]
    fn data_integrity_u8_test() {
        let vec: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2].into_iter().map(Rc::new).collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u8>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2, 52, 1, 3, 41, 23, 2]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u8>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i16_test() {
        let vec: Vec<Rc<i16>> = vec![1i16, -3, 1002, -231, 32]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i16>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<i16>> = vec![1i16, -3, 1002, -231, 32, 42, -123, 4]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i16>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i32_test() {
        let vec: Vec<Rc<i32>> = vec![1i32, -3, 1002, -231, 32]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<i32>> = vec![1i32, -3, 1002, -231, 32, 42, -123]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Foo {
        a: u8,
        b: i64,
    }

    #[test]
    fn from_empty_vec_test() {
        let vec: Vec<Rc<u32>> = Vec::new();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<String>> = Vec::new();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<String>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<Foo>> = Vec::new();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<Foo>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn from_struct_test() {
        let f1 = Foo { a: 3, b: -32 };
        let f2 = Foo {
            a: 33,
            b: -3342432412,
        };
        let vec: Vec<Rc<Foo>> = vec![Rc::new(f1.clone()), Rc::new(f2.clone())];
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        assert_eq!(Rc::new(f1), buf.get_ref_as::<Rc<Foo>>(0).unwrap().clone());
        assert_eq!(Rc::new(f2), buf.get_ref_as::<Rc<Foo>>(1).unwrap().clone());
        let nu_vec: Vec<Rc<Foo>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn from_strings_test() {
        let vec: Vec<Rc<String>> = vec![
            String::from("hi"),
            String::from("hello"),
            String::from("goodbye"),
            String::from("bye"),
            String::from("supercalifragilisticexpialidocious"),
            String::from("42"),
        ]
        .into_iter()
        .map(Rc::new)
        .collect();
        let buf = VecDropAll::from(vec.clone()); // Convert into buffer
        assert_eq!(
            &Rc::new("hi".to_string()),
            buf.get_ref_as::<Rc<String>>(0).unwrap()
        );
        assert_eq!(
            &Rc::new("hello".to_string()),
            buf.get_ref_as::<Rc<String>>(1).unwrap()
        );
        assert_eq!(
            &Rc::new("goodbye".to_string()),
            buf.get_ref_as::<Rc<String>>(2).unwrap()
        );
        let nu_vec: Vec<Rc<String>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn iter_test() {
        let vec_u8: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2, 4, 128, 32]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecDropAll::from(vec_u8.clone()); // Convert into buffer
        for (i, val) in buf.iter_as::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }
    }

    #[test]
    fn large_sizes_clone() {
        for i in 100000..100010 {
            let vec: Vec<Rc<u8>> = vec![32u8; i].into_iter().map(Rc::new).collect();
            let buf = VecDropAll::from(vec.clone()); // Convert into buffer
            let nu_vec: Vec<Rc<u8>> = buf.into_vec().unwrap(); // Convert back into vec
            assert_eq!(vec, nu_vec);
        }
    }

    /// This test checks that an error is returned whenever the user tries to access data with the
    /// wrong type data.
    #[test]
    fn wrong_type_test() {
        let vec: Vec<Rc<u8>> = vec![1, 23, 2, 42, 11].into_iter().map(Rc::new).collect();
        let mut buf = VecDropAll::from(vec.clone()); // Convert into buffer
        assert_eq!(vec, buf.clone_into_vec::<Rc<u8>>().unwrap());

        assert!(buf.clone_into_vec::<Rc<f64>>().is_none());
        assert!(buf.as_slice_as::<Rc<f64>>().is_none());
        assert!(buf.iter_as::<Rc<[u8; 3]>>().is_none());
        assert!(buf.get_ref_as::<Rc<i32>>(1).is_none());
        assert!(buf.get_mut_as::<Rc<i32>>(2).is_none());
    }

    /// Test pushing values and bytes to a buffer.
    #[test]
    fn push_test() {
        let mut vec_u8: Vec<Rc<u8>> = vec![1u8, 23, 2].into_iter().map(Rc::new).collect();
        let mut buf = VecDropAll::from(vec_u8.clone()); // Convert into buffer
        for (i, val) in buf.iter_as::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }

        vec_u8.push(Rc::new(42u8));
        buf.push_as(Rc::new(42u8)).unwrap(); // must provide explicit type

        for (i, val) in buf.iter_as::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }

        vec_u8.push(Rc::new(11u8));
        buf.push_as(Rc::new(11u8)).unwrap();

        for (i, val) in buf.iter_as::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }
    }

    /// Test appending to a buffer from another buffer.
    #[test]
    fn append_test() {
        let mut buf = VecDropAll::with_type::<Rc<u8>>(); // Create an empty buffer.

        let data: Vec<Rc<u8>> = vec![1, 23, 2, 42, 11].into_iter().map(Rc::new).collect();
        // Append an ordianry vector of data.
        let mut other_buf = VecDropAll::from_vec(data.clone());
        buf.append(&mut other_buf);

        assert!(other_buf.is_empty());

        for (i, val) in buf.iter_as::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &data[i]);
        }
    }

    #[test]
    fn dynamic_vtables_assignment() {
        use crate::{from_dyn, into_dyn};

        let buf = VecDropAll::with_type::<u8>(); // Create an empty buffer.

        let mut buf_dyn = into_dyn![VecDrop<dyn HasAllTrait>](buf);

        buf_dyn.push(BoxValue::<AllTraitVTable>::new(1u8));
        buf_dyn.push(BoxValue::<AllTraitVTable>::new(100u8));
        buf_dyn.push(BoxValue::<AllTraitVTable>::new(23u8));

        let buf = from_dyn![VecDrop<dyn HasAllTrait as AllTraitVTable>](buf_dyn);
        let vec: Vec<u8> = buf.into_vec().unwrap();

        assert_eq!(vec, vec![1u8, 100, 23]);
    }

    // This test checks that cloning and dropping clones works correctly.
    #[test]
    fn clone_test() {
        let buf = VecDropAll::with_size::<Rc<u8>>(3, Rc::new(1u8));
        assert_eq!(&buf, &buf.clone());
    }

    #[cfg(feature = "numeric")]
    #[test]
    fn convert_float_test() {
        let vecf64 = vec![1f64, -3.0, 10.02, -23.1, 32e-1];
        let buf: VecDrop = VecDrop::from(vecf64.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.cast_into_vec().unwrap(); // Convert back into vec
        let vecf32 = vec![1f32, -3.0, 10.02, -23.1, 32e-1];
        assert_eq!(vecf32, nu_vec);

        let buf: VecDrop = VecDrop::from(vecf32.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.cast_into_vec().unwrap(); // Convert back into vec
        for (&a, &b) in vecf64.iter().zip(nu_vec.iter()) {
            assert!((a - b).abs() < 1e-6f64 * f64::max(a, b).abs());
        }

        let vecf64 = vec![1f64, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        let buf: VecDrop = VecDrop::from(vecf64.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.cast_into_vec().unwrap(); // Convert back into vec
        let vecf32 = vec![1f32, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        assert_eq!(vecf32, nu_vec);
        let buf: VecDrop = VecDrop::from(vecf32.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.cast_into_vec().unwrap(); // Convert back into vec
        for (&a, &b) in vecf64.iter().zip(nu_vec.iter()) {
            assert!((a - b).abs() < 1e-6 * f64::max(a, b).abs());
        }
    }
}
