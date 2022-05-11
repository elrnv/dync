//! This module defines a typeless homogeneous vector data structure optimized to be written to and
//! read from standard `Vec`s. It is not unlike `Vec<dyn Trait>` but stores only a single vtable
//! for all the values in the vector producing better data locality.
//!
//! [`VecCopy`] is particularly useful when dealing with plain data whose type is determined at
//! run time.  Note that data is stored in the underlying byte vectors in native endian form,
//! endianness is not handled by this type.
//!
//! # Caveats
//!
//! [`VecCopy`] doesn't support zero-sized types.
//!
//! [`VecCopy`]: struct.VecCopy

use std::{
    any::{Any, TypeId},
    mem::{size_of, MaybeUninit},
    slice,
};

// At the time of this writing, there is no evidence that there is a
// significant benefit in sharing vtables via Rc or Arc, but to make potential
// future refactoring easier we use the Ptr alias.
use std::boxed::Box as Ptr;

#[cfg(feature = "numeric")]
use std::fmt;

#[cfg(feature = "numeric")]
use num_traits::{cast, NumCast, Zero};

use crate::copy_value::*;
use crate::elem::*;
use crate::slice_copy::*;
use crate::vec_void::*;
use crate::vtable::*;
use crate::{ElementBytes, ElementBytesMut};

/// Buffer of untyped `Copy` values.
///
/// `VecCopy` keeps track of the type stored within via an explicit `TypeId` member. This allows
/// one to hide the type from the compiler and check it only when necessary. It is particularly
/// useful when the type of data is determined at runtime (e.g. when parsing numeric data).
///
/// # Safety
///
/// The data representing a type is never interpreted as anything
/// other than a type with an identical `TypeId`, which are assumed to have an
/// identical memory layout throughout the execution of the program.
///
/// It is an error to share this type between independently compiled binaries since `TypeId`s
/// are not stable, and thus reinterpreting the values may not work as expected.
#[derive(Clone)]
pub struct VecCopy<V = ()>
where
    V: ?Sized,
{
    /// Raw data.
    pub(crate) data: VecVoid,

    /// VTable pointer.
    pub(crate) vtable: Ptr<V>,
}

impl<V> VecCopy<V> {
    /// Construct an empty `VecCopy` with a specific type.
    #[inline]
    pub fn with_type<T: CopyElem>() -> Self
    where
        V: VTable<T>,
    {
        Self::from_vec(Vec::new())
    }

    /// It is unsafe to construct a `VecCopy` if `T` is not a `CopyElem`.
    #[cfg(feature = "traits")]
    #[inline]
    pub(crate) unsafe fn with_type_non_copy<T: Any>() -> Self
    where
        V: VTable<T>,
    {
        Self::from_vec_non_copy(Vec::new())
    }

    /// Construct an empty `VecCopy` with a capacity for a given number of typed elements. For
    /// setting byte capacity use `with_byte_capacity`.
    #[inline]
    pub fn with_capacity<T: CopyElem>(n: usize) -> Self
    where
        V: VTable<T>,
    {
        Self::from_vec(Vec::with_capacity(n))
    }

    /// It is unsafe to construct a `VecCopy` if `T` is not `Copy`.
    #[cfg(feature = "traits")]
    #[inline]
    pub(crate) unsafe fn with_capacity_non_copy<T: Any>(n: usize) -> Self
    where
        V: VTable<T>,
    {
        Self::from_vec_non_copy(Vec::with_capacity(n))
    }

    /// Construct a typed `VecCopy` with a given size and filled with the specified default
    /// value.
    ///
    /// #  Examples
    /// ```
    /// use dync::VecCopy;
    /// let buf: VecCopy = VecCopy::with_size(8, 42usize); // Create buffer
    /// let buf_vec: Vec<usize> = buf.into_vec().unwrap(); // Convert into `Vec`
    /// assert_eq!(buf_vec, vec![42usize; 8]);
    /// ```
    #[inline]
    pub fn with_size<T: CopyElem>(n: usize, def: T) -> Self
    where
        V: VTable<T>,
    {
        Self::from_vec(vec![def; n])
    }

    /// Construct a `VecCopy` from a given `Vec<T>` reusing the space already allocated by the
    /// given vector.
    ///
    /// #  Examples
    /// ```
    /// use dync::VecCopy;
    /// let vec = vec![1u8, 3, 4, 1, 2];
    /// let buf: VecCopy = VecCopy::from_vec(vec.clone()); // Convert into buffer
    /// let nu_vec: Vec<u8> = buf.into_vec().unwrap(); // Convert back into `Vec`
    /// assert_eq!(vec, nu_vec);
    /// ```
    pub fn from_vec<T: CopyElem>(vec: Vec<T>) -> Self
    where
        V: VTable<T>,
    {
        // SAFETY: `T` is a `CopyElem`.
        unsafe { Self::from_vec_non_copy(vec) }
    }

    /// It is unsafe to call this for `T` that is not a `CopyElem`.
    pub(crate) unsafe fn from_vec_non_copy<T: Any>(vec: Vec<T>) -> Self
    where
        V: VTable<T>,
    {
        assert_ne!(
            size_of::<T>(),
            0,
            "VecCopy doesn't support zero sized types."
        );

        VecCopy {
            data: VecVoid::from_vec(vec),
            vtable: Ptr::new(V::build_vtable()),
        }
    }

    /// Construct a `VecCopy` from a given slice by copying the data.
    #[inline]
    pub fn from_slice<T: CopyElem>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        let mut vec = Vec::with_capacity(slice.len());
        vec.extend_from_slice(slice);
        Self::from_vec(vec)
    }

    /// It is unsafe to call this for `T` that is not a `CopyElem`.
    #[cfg(feature = "traits")]
    #[inline]
    pub(crate) unsafe fn from_slice_non_copy<T: Any + Clone>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        let mut vec = Vec::with_capacity(slice.len());
        vec.extend_from_slice(slice);
        Self::from_vec_non_copy(vec)
    }
}

impl<V: ?Sized> VecCopy<V> {
    /// Construct a `VecCopy` with the same type as the given buffer without copying its data.
    #[inline]
    pub fn with_type_from<'a>(other: impl Into<crate::meta::Meta<VTableRef<'a, V>>>) -> Self
    where
        V: Clone + 'a,
    {
        let mut other = other.into();

        fn new<T: 'static>(elem: &mut ElemInfo) -> VecVoid {
            unsafe { VecVoid::from_vec_override(Vec::<T>::with_capacity(elem.size), *elem) }
        }

        VecCopy {
            data: eval_align!(other.elem.alignment; new::<_>(&mut other.elem)),
            vtable: other.vtable.into_owned(),
        }
    }

    /// Construct a `SliceCopy` from raw bytes and type metadata.
    ///
    /// # Safety
    ///
    /// Almost exclusively the only inputs that are safe here are the ones returned by
    /// `into_raw_parts`.
    ///
    /// This function should not be used other than in internal APIs. It exists to enable the
    /// `into_dyn` macro until `CoerceUsize` is stabilized.
    #[inline]
    pub unsafe fn from_raw_parts(data: VecVoid, vtable: Ptr<V>) -> VecCopy<V> {
        VecCopy { data, vtable }
    }

    /// Convert this collection into its raw components.
    ///
    /// This function exists mainly to enable the `into_dyn` macro until `CoerceUnsized` is
    /// stabilized.
    #[inline]
    pub fn into_raw_parts(self) -> (VecVoid, Ptr<V>) {
        let VecCopy { data, vtable } = self;
        (data, vtable)
    }

    /// Upcast the `VecCopy` into a more general base `VecCopy`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> VecCopy<U>
    where
        V: Clone,
    {
        self.upcast_with(U::from)
    }

    // Helper for upcasts
    #[inline]
    pub fn upcast_with<U>(self, f: impl FnOnce(V) -> U) -> VecCopy<U>
    where
        V: Clone,
    {
        VecCopy {
            data: self.data,
            vtable: Ptr::new(f((*self.vtable).clone())),
        }
    }

    /// Reserve `additional` extra elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional)
    }

    /// Resizes the buffer in-place to store `new_len` elements and returns an optional
    /// mutable reference to `Self`.
    ///
    /// If `T` does not correspond to the underlying element type, then `None` is returned and the
    /// `VecCopy` is left unchanged.
    ///
    /// This function has the similar properties to `Vec::resize`.
    #[inline]
    pub fn resize<T: CopyElem>(&mut self, new_len: usize, value: T) -> Option<&mut Self> {
        self.check_ref::<T>()?;
        if new_len >= self.len() {
            let diff = new_len - self.len();
            self.data.reserve(diff);
            for _ in 0..diff {
                self.push_as(value);
            }
        } else {
            self.data.truncate(new_len);
        }
        Some(self)
    }

    /// Copy data from a given slice into the current buffer.
    ///
    /// The `VecCopy` is extended if the given slice is larger than the number of elements
    /// already stored in this `VecCopy`.
    #[inline]
    pub fn copy_from_slice<T: CopyElem>(&mut self, slice: &[T]) -> Option<&mut Self> {
        let mut this_slice = self.as_mut_slice();
        match this_slice.copy_from_slice(slice) {
            Some(_) => Some(self),
            None => None,
        }
    }

    /// Clear the data buffer without destroying its type information.
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Fill the current buffer with copies of the given value. The size of the buffer is left
    /// unchanged. If the given type doesn't patch the internal type, `None` is returned, otherwise
    /// a mut reference to the modified buffer is returned.
    ///
    /// #  Examples
    /// ```
    /// use dync::VecCopy;
    /// let vec = vec![1u8, 3, 4, 1, 2];
    /// let mut buf: VecCopy = VecCopy::from_vec(vec.clone()); // Convert into buffer
    /// buf.fill(0u8);
    /// assert_eq!(buf.into_vec::<u8>().unwrap(), vec![0u8, 0, 0, 0, 0]);
    /// ```
    #[inline]
    pub fn fill<T: CopyElem>(&mut self, def: T) -> Option<&mut Self> {
        for v in self.iter_mut_as::<T>()? {
            *v = def;
        }
        Some(self)
    }

    /// Add an element to this buffer.
    ///
    /// If the type of the given element coincides with the type
    /// stored by this buffer, then the modified buffer is returned via a mutable reference.
    /// Otherwise, `None` is returned.
    #[inline]
    pub fn push_as<T: Any>(&mut self, element: T) -> Option<&mut Self> {
        self.check_mut::<T>().map(|s| {
            // SAFETY: Checked using `check_mut`.
            unsafe {
                s.data.apply_unchecked(|v| {
                    v.push(element);
                });
            }
            s
        })
    }

    /// Check if the current buffer contains elements of the specified type. Returns `Some(self)`
    /// if the type matches and `None` otherwise.
    #[inline]
    pub fn check<T: Any>(self) -> Option<Self> {
        if TypeId::of::<T>() != self.element_type_id() {
            None
        } else {
            Some(self)
        }
    }

    /// Check if the current buffer contains elements of the specified type. Returns `None` if the
    /// check fails, otherwise a reference to self is returned.
    #[inline]
    pub fn check_ref<T: Any>(&self) -> Option<&Self> {
        if TypeId::of::<T>() != self.element_type_id() {
            None
        } else {
            Some(self)
        }
    }

    /// Check if the current buffer contains elements of the specified type. Same as `check_ref`
    /// but consumes and produces a mut reference to self.
    #[inline]
    pub fn check_mut<T: Any>(&mut self) -> Option<&mut Self> {
        if TypeId::of::<T>() != self.element_type_id() {
            None
        } else {
            Some(self)
        }
    }

    /*
     * Accessors
     */

    /// Get the `TypeId` of data stored within this buffer.
    #[inline]
    pub fn element_type_id(&self) -> TypeId {
        self.data.elem.type_id
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

    /// Get the capacity of this buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Return an iterator to a slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    ///
    /// # Examples
    /// ```
    /// use dync::VecCopy;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf: VecCopy = VecCopy::from(vec.clone()); // Convert into buffer
    /// for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
    ///     assert_eq!(val, vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&self) -> Option<slice::Iter<T>> {
        self.as_slice_as::<T>().map(|x| x.iter())
    }

    /// Return an iterator to a mutable slice representing typed data.
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn iter_mut_as<T: Any>(&mut self) -> Option<slice::IterMut<T>> {
        self.as_mut_slice_as::<T>().map(|x| x.iter_mut())
    }

    /// Append copied items from this buffer to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise.
    #[inline]
    pub fn append_copy_to_vec<'a, T: CopyElem>(
        &self,
        vec: &'a mut Vec<T>,
    ) -> Option<&'a mut Vec<T>> {
        let iter = self.iter_as()?;
        // Allocate only after we know the type is right to prevent unnecessary allocations.
        vec.reserve(self.len());
        vec.extend(iter);
        Some(vec)
    }

    /// Copies contents of `self` into the given `Vec`.
    #[inline]
    pub fn copy_into_vec<T: CopyElem>(&self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        match self.append_copy_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }

    /// Convert this vector in to a `Vec<T>`.
    ///
    /// This is like using the `Into` trait, but it helps the compiler
    /// determine the type `T` automatically.
    ///
    /// This function returns `None` if `T` is not the same as the `T` that
    /// this vector was created with.
    #[inline]
    pub fn into_vec<T: Any>(self) -> Option<Vec<T>> {
        // SAFETY: `T` is `CopyElem` guaranteed at construction.
        self.check::<T>()
            .map(|s| unsafe { s.data.into_vec_unchecked::<T>() })
    }

    /// Convert this buffer into a typed slice.
    ///
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice_as<T: Any>(&self) -> Option<&[T]> {
        let ptr = self.check_ref::<T>()?.data.ptr as *const T;
        // SAFETY: `T` is `CopyElem` guaranteed at construction.
        Some(unsafe { slice::from_raw_parts(ptr, self.len()) })
    }

    /// Convert this buffer into a typed slice.
    ///
    /// # Safety
    ///
    /// The underlying type must correspond to `T`.
    #[cfg(feature = "numeric")]
    #[inline]
    pub(crate) unsafe fn as_slice_as_unchecked<T: Any>(&self) -> &[T] {
        slice::from_raw_parts(self.data.ptr as *const T, self.len())
    }

    /// Convert this buffer into a typed mutable slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_mut_slice_as<T: Any>(&mut self) -> Option<&mut [T]> {
        let ptr = self.check_mut::<T>()?.data.ptr as *mut T;
        // SAFETY: `T` is `CopyElem` guaranteed at construction.
        Some(unsafe { slice::from_raw_parts_mut(ptr, self.len()) })
    }

    /// Get `i`'th element of the buffer by value.
    #[inline]
    pub fn get_as<T: CopyElem>(&self, i: usize) -> Option<T> {
        assert!(i < self.len());
        let ptr = self.check_ref::<T>()?.data.ptr as *const T;
        // SAFETY: `T` is `CopyElem` guaranteed at construction.
        Some(unsafe { *ptr.add(i) })
    }

    /// Get a `const` reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_ref_as<T: Any>(&self, i: usize) -> Option<&T> {
        assert!(i < self.len());
        let ptr = self.check_ref::<T>()?.data.ptr as *const T;
        // SAFETY: `T` is `CopyElem` guaranteed at construction.
        Some(unsafe { &*ptr.add(i) })
    }

    /// Get a mutable reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_mut_as<T: Any>(&mut self, i: usize) -> Option<&mut T> {
        assert!(i < self.len());
        let ptr = self.check_mut::<T>()?.data.ptr as *mut T;
        // SAFETY: `T` is `CopyElem` guaranteed at construction.
        Some(unsafe { &mut *ptr.add(i) })
    }

    /// Move elements from `buf` to this buffer.
    ///
    /// The given buffer must have the same underlying type as `self`.
    #[inline]
    pub fn append(&mut self, buf: &mut VecCopy<V>) -> Option<&mut Self> {
        if buf.element_type_id() == self.element_type_id() {
            self.data.append(&mut buf.data);
            Some(self)
        } else {
            None
        }
    }

    /// Rotates the slice in-place such that the first `mid` elements of the slice move to the end
    /// while the last `self.len() - mid` elements move to the front. After calling `rotate_left`,
    /// the element previously at index `mid` will become the first element in the slice.
    ///
    /// # Example
    ///
    /// ```
    /// use dync::*;
    /// let mut buf: VecCopy = VecCopy::from_vec(vec![1u32,2,3,4,5]);
    /// buf.rotate_left(3);
    /// assert_eq!(buf.as_slice_as::<u32>().unwrap(), &[4,5,1,2,3]);
    /// ```
    #[inline]
    pub fn rotate_left(&mut self, mid: usize) {
        self.data.rotate_left(mid);
    }

    /// Rotates the slice in-place such that the first `self.len() - k` elements of the slice move
    /// to the end while the last `k` elements move to the front. After calling `rotate_right`, the
    /// element previously at index `k` will become the first element in the slice.
    ///
    /// # Example
    ///
    /// ```
    /// use dync::*;
    /// let mut buf: VecCopy = VecCopy::from_vec(vec![1u32,2,3,4,5]);
    /// buf.rotate_right(3);
    /// assert_eq!(buf.as_slice_as::<u32>().unwrap(), &[3,4,5,1,2]);
    /// ```
    #[inline]
    pub fn rotate_right(&mut self, k: usize) {
        self.data.rotate_right(k);
    }

    /*
     * Value API. This allows users to manipulate contained data without knowing the element type.
     */

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get_ref(&self, i: usize) -> CopyValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        let num_bytes = self.data.elem.num_bytes();
        unsafe {
            CopyValueRef::from_raw_parts(
                std::slice::from_raw_parts(
                    (self.data.ptr as *const T0).add(i * num_bytes) as *const MaybeUninit<u8>,
                    num_bytes,
                ),
                self.element_type_id(),
                self.data.elem.alignment,
                self.vtable.as_ref(),
            )
        }
    }

    /// Get a mutable reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> CopyValueMut<V> {
        debug_assert!(i < self.len());
        let num_bytes = self.data.elem.num_bytes();
        unsafe {
            CopyValueMut::from_raw_parts(
                std::slice::from_raw_parts_mut(
                    (self.data.ptr as *mut u8).add(i * num_bytes) as *mut MaybeUninit<u8>,
                    num_bytes,
                ),
                self.element_type_id(),
                self.data.elem.alignment,
                self.vtable.as_ref(),
            )
        }
    }

    /// Return an iterator over untyped value references stored in this buffer.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    ///
    /// # Examples
    /// ```
    /// use dync::VecCopy;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf: VecCopy = VecCopy::from(vec.clone()); // Convert into buffer
    /// for (i, val) in buf.iter().enumerate() {
    ///     assert_eq!(val.downcast::<f32>().unwrap(), &vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = CopyValueRef<V>> {
        self.data.byte_chunks().map(move |bytes| unsafe {
            CopyValueRef::from_raw_parts(
                bytes,
                self.element_type_id(),
                self.data.elem.alignment,
                &*self.vtable,
            )
        })
    }

    /// Return an iterator over untyped value references stored in this buffer.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    ///
    /// # Examples
    /// ```
    /// use dync::*;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let mut buf: VecCopy = VecCopy::from(vec.clone()); // Convert into buffer
    /// for (i, val) in buf.iter_mut().enumerate() {
    ///     val.copy(CopyValueRef::new(&100.0f32));
    /// }
    /// assert_eq!(buf.into_vec::<f32>().unwrap(), vec![100.0f32; 5]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = CopyValueMut<V>> {
        let &mut VecCopy {
            ref mut data,
            ref vtable,
        } = self;
        let element_type_id = data.elem.type_id;
        let element_alignment = data.elem.alignment;
        let vtable = vtable.as_ref();
        unsafe {
            data.byte_chunks_mut().map(move |bytes| {
                CopyValueMut::from_raw_parts(bytes, element_type_id, element_alignment, vtable)
            })
        }
    }

    /// Push a value to this `VecCopy` by reference and return a mutable reference to `Self`.
    ///
    /// If the type of the value doesn't match the internal element type, return `None`.
    ///
    /// Note that it is not necessary for vtables of the value and this vector to match. IF the
    /// types coincide, we know that either of the vtables is valid, so we just stick with the one
    /// we already have in the container.
    ///
    /// # Panics
    ///
    /// This function panics if the size of the given value doesn't match the size of the stored
    /// value.
    #[inline]
    pub fn push<U>(&mut self, value: CopyValueRef<U>) -> Option<&mut Self> {
        assert_eq!(value.size(), self.element_size());
        if value.value_type_id() == self.element_type_id() {
            self.data.push(value.bytes);
            Some(self)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_slice(&self) -> SliceCopy<V> {
        let &VecCopy {
            ref data,
            ref vtable,
        } = self;
        let num_elem_bytes = data.elem.num_bytes();
        unsafe {
            let slice = std::slice::from_raw_parts(
                data.ptr as *const MaybeUninit<u8>,
                data.len * num_elem_bytes,
            );
            SliceCopy::from_raw_parts(slice, data.elem, vtable.as_ref())
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> SliceCopyMut<V> {
        let &mut VecCopy {
            ref mut data,
            ref vtable,
        } = self;
        let num_elem_bytes = data.elem.num_bytes();
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                data.ptr as *mut MaybeUninit<u8>,
                data.len * num_elem_bytes,
            );
            SliceCopyMut::from_raw_parts(slice, data.elem, vtable.as_ref())
        }
    }
}

impl<V> VecCopy<V> {
    /*
     * Methods specific to buffers storing numeric data
     */

    #[cfg(feature = "numeric")]
    /// Cast a numeric `VecCopy` into the given output `Vec` type.
    pub fn cast_into_vec<T>(&self) -> Vec<T>
    where
        T: CopyElem + NumCast + Zero,
    {
        // Helper function (generic on the input) to convert the given VecCopy into Vec.
        unsafe fn convert_into_vec<I, O, V>(buf: &VecCopy<V>) -> Vec<O>
        where
            I: CopyElem + Any + NumCast,
            O: CopyElem + NumCast + Zero,
        {
            debug_assert_eq!(buf.element_type_id(), TypeId::of::<I>()); // Check invariant.
            buf.as_slice_as_unchecked()
                .iter()
                .map(|elem: &I| cast(*elem).unwrap_or_else(O::zero))
                .collect()
        }
        call_numeric_buffer_fn!( convert_into_vec::<_, T, V>(&self) or { Vec::new() } )
    }

    #[cfg(feature = "numeric")]
    /// Display the contents of this buffer reinterpreted in the given type.
    unsafe fn reinterpret_display<T: CopyElem + fmt::Display>(&self, f: &mut fmt::Formatter) {
        debug_assert_eq!(self.element_type_id(), TypeId::of::<T>()); // Check invariant.
        for item in self.as_slice_as_unchecked::<T>().iter() {
            write!(f, "{} ", item).expect("Error occurred while writing an VecCopy.");
        }
    }
}

impl<'a, V: Clone + ?Sized + 'a> std::iter::FromIterator<CopyValueRef<'a, V>> for VecCopy<V> {
    /// Construct a `VecCopy` type from a *non-empty* iterator.
    ///
    /// # Panics
    ///
    /// This function will panic if the given iterator is empty.
    /// This is because we don't know the element types until we see one since
    /// the types are erased in both `CopyValueRef` and `VecCopy`.
    #[inline]
    fn from_iter<T: IntoIterator<Item = CopyValueRef<'a, V>>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let next = iter
            .next()
            .expect("VecCopy cannot be built from an empty untyped iterator.");
        let mut data = Self::with_type_from(next.clone());
        data.push(next);
        data.extend(iter);
        data
    }
}

impl<'a, V: ?Sized + 'a> Extend<CopyValueRef<'a, V>> for VecCopy<V> {
    #[inline]
    fn extend<T: IntoIterator<Item = CopyValueRef<'a, V>>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for value in iter {
            assert_eq!(value.size(), self.element_size());
            assert_eq!(value.value_type_id(), self.element_type_id());
            self.data.push(value.bytes);
        }
    }
}

/*
 * Advanced methods to probe buffer internals.
 */

impl<V: ?Sized + Clone> VecCopy<V> {
    /// Clones this `VecCopy` using the given function.
    #[cfg(feature = "traits")]
    pub(crate) fn clone_with(&self, clone: impl FnOnce(&VecVoid) -> VecVoid) -> Self {
        VecCopy {
            data: clone(&self.data),
            vtable: Ptr::clone(&self.vtable),
        }
    }
}

impl<V: ?Sized> VecCopy<V> {
    /// Get a `const` reference to the `i`'th element of the buffer.
    ///
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current buffer. See the implementation for details.
    ///
    /// # Safety
    ///
    /// It is assumed that that the buffer contains elements of type `T` and that `i` is strictly
    /// less than the length of this vector, otherwise this function will cause undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked_ref<T: Any>(&self, i: usize) -> &T {
        let ptr = self.data.ptr as *const T;
        &*ptr.add(i)
    }

    /// Get a mutable reference to the `i`'th element of the buffer.
    ///
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current buffer. See the implementation for details.
    ///
    /// # Safety
    ///
    /// It is assumed that that the buffer contains elements of type `T` and that `i` is strictly
    /// less than the length of this vector, otherwise this function will cause undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: Any>(&mut self, i: usize) -> &mut T {
        let ptr = self.data.ptr as *mut T;
        &mut *ptr.add(i)
    }
}

impl VecVoid {
    /// Iterate over chunks type sized chunks of bytes without interpreting them.
    ///
    /// This avoids needing to know what type data you're dealing with. This type of iterator is
    /// useful for transferring data from one place to another for a generic buffer.
    #[inline]
    pub fn byte_chunks<'a>(&'a self) -> impl Iterator<Item = &'a [MaybeUninit<u8>]> + 'a {
        let chunk_size = self.elem.num_bytes();
        self.bytes().chunks_exact(chunk_size)
    }

    /// Mutably iterate over chunks type sized chunks of bytes without interpreting them. This
    /// avoids needing to know what type data you're dealing with. This type of iterator is useful
    /// for transferring data from one place to another for a generic buffer, or modifying the
    /// underlying untyped bytes (e.g. bit twiddling).
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe since the returned bytes may be modified
    /// arbitrarily, which may potentially produce malformed values.
    #[inline]
    pub unsafe fn byte_chunks_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut [MaybeUninit<u8>]> + 'a {
        let chunk_size = self.elem.num_bytes();
        let slice = std::slice::from_raw_parts_mut(
            self.ptr as *mut MaybeUninit<u8>,
            chunk_size * self.len(),
        );
        slice.chunks_exact_mut(chunk_size)
    }

    /// Get a `const` reference to the byte slice of the `i`'th element of the buffer.
    #[inline]
    pub fn get_bytes(&self, i: usize) -> &[MaybeUninit<u8>] {
        debug_assert!(i < self.len());
        let element_size = self.element_size();
        &self.bytes()[i * element_size..(i + 1) * element_size]
    }

    /// Get a mutable reference to the byte slice of the `i`'th element of the buffer.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe since the returned bytes may be modified
    /// arbitrarily, which may potentially produce malformed values.
    #[inline]
    pub unsafe fn get_bytes_mut(&mut self, i: usize) -> &mut [MaybeUninit<u8>] {
        debug_assert!(i < self.len());
        self.index_byte_slice_mut(i)
    }
}

impl ElementBytes for VecVoid {
    fn element_size(&self) -> usize {
        self.elem.num_bytes()
    }
    fn bytes(&self) -> &[MaybeUninit<u8>] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr as *const MaybeUninit<u8>,
                self.len * self.elem.num_bytes(),
            )
        }
    }
}

impl ElementBytesMut for VecVoid {
    unsafe fn bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        std::slice::from_raw_parts_mut(
            self.ptr as *mut MaybeUninit<u8>,
            self.len * self.elem.num_bytes(),
        )
    }
}

impl<V: ?Sized> ElementBytes for VecCopy<V> {
    fn element_size(&self) -> usize {
        self.data.element_size()
    }
    fn bytes(&self) -> &[MaybeUninit<u8>] {
        self.data.bytes()
    }
}

impl<V: ?Sized> ElementBytesMut for VecCopy<V> {
    unsafe fn bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.data.bytes_mut()
    }
}

/// Convert a `Vec<T>` to a `VecCopy`.
impl<T, V> From<Vec<T>> for VecCopy<V>
where
    T: CopyElem,
    V: VTable<T>,
{
    #[inline]
    fn from(vec: Vec<T>) -> VecCopy<V> {
        VecCopy::from_vec(vec)
    }
}

/// Convert a `&[T]` to a `VecCopy`.
impl<'a, T, V> From<&'a [T]> for VecCopy<V>
where
    T: CopyElem,
    V: VTable<T>,
{
    #[inline]
    fn from(slice: &'a [T]) -> VecCopy<V> {
        VecCopy::from_slice(slice)
    }
}

/// Convert a `VecCopy` to a `Option<Vec<T>>`.
impl<T, V: ?Sized> Into<Option<Vec<T>>> for VecCopy<V>
where
    T: CopyElem,
{
    #[inline]
    fn into(self) -> Option<Vec<T>> {
        self.into_vec()
    }
}

#[cfg(feature = "numeric")]
/// Implement pretty printing of numeric `VecCopy` data.
impl<V> fmt::Display for VecCopy<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        call_numeric_buffer_fn!( self.reinterpret_display::<_>(f) or {
            println!("Unknown VecCopy type for pretty printing.");
        } );
        write!(f, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type VecUnit = super::VecCopy<()>;

    /// Test various ways to create a data buffer.
    #[test]
    fn initialization_test() {
        // Empty typed buffer.
        let a = VecUnit::with_type::<f32>();
        assert_eq!(a.len(), 0);
        assert_eq!(a.element_type_id(), TypeId::of::<f32>());

        // Empty buffer typed by the given type id.
        #[cfg(feature = "traits")]
        {
            let b = VecUnit::with_type_from(&a);
            assert_eq!(b.len(), 0);
            assert_eq!(b.element_type_id(), TypeId::of::<f32>());
        }

        // Empty typed buffer with a given capacity.
        let a = VecUnit::with_capacity::<f32>(4);
        assert_eq!(a.len(), 0);
        assert_eq!(a.element_type_id(), TypeId::of::<f32>());
    }

    /// Test resizing a buffer.
    #[test]
    fn resize() {
        let mut a = VecUnit::with_type::<f32>();

        // Increase the size of a.
        a.resize(3, 1.0f32);

        assert_eq!(a.len(), 3);
        for i in 0..3 {
            assert_eq!(a.get_as::<f32>(i).unwrap(), 1.0f32);
        }

        // Truncate a.
        a.resize(2, 1.0f32);

        assert_eq!(a.len(), 2);
        for i in 0..2 {
            assert_eq!(a.get_as::<f32>(i).unwrap(), 1.0f32);
        }
    }

    #[test]
    #[should_panic]
    fn zero_size_with_type_test() {
        let _a = VecUnit::with_type::<()>();
    }

    #[test]
    #[should_panic]
    fn zero_size_with_capacity_test() {
        let _a = VecUnit::with_capacity::<()>(2);
    }

    #[test]
    #[should_panic]
    fn zero_size_from_vec_test() {
        let _a = VecUnit::from_vec(vec![(); 3]);
    }

    #[test]
    #[should_panic]
    fn zero_size_with_size_test() {
        let _a = VecUnit::with_size(3, ());
    }

    #[test]
    #[should_panic]
    fn zero_size_from_slice_test() {
        let v = vec![(); 3];
        let _a = VecUnit::from_slice(&v);
    }

    #[test]
    //#[should_panic]
    fn zero_size_copy_from_slice_test() {
        let v = vec![(); 3];
        let mut a = VecUnit::with_size(0, 1i32);
        a.copy_from_slice(&v);
    }

    #[test]
    fn data_integrity_u8_test() {
        let vec = vec![1u8, 3, 4, 1, 2];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<u8> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1u8, 3, 4, 1, 2, 52, 1, 3, 41, 23, 2];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<u8> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i16_test() {
        let vec = vec![1i16, -3, 1002, -231, 32];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i16> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1i16, -3, 1002, -231, 32, 42, -123, 4];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i16> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i32_test() {
        let vec = vec![1i32, -3, 1002, -231, 32];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1i32, -3, 1002, -231, 32, 42, -123];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_f32_test() {
        let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43, 2e-1];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_f64_test() {
        let vec = vec![1f64, -3.0, 10.02, -23.1, 32e-1];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1f64, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[cfg(feature = "numeric")]
    #[test]
    fn convert_float_test() {
        let vecf64 = vec![1f64, -3.0, 10.02, -23.1, 32e-1];
        let buf = VecUnit::from(vecf64.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.cast_into_vec(); // Convert back into vec
        let vecf32 = vec![1f32, -3.0, 10.02, -23.1, 32e-1];
        assert_eq!(vecf32, nu_vec);

        let buf = VecUnit::from(vecf32.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.cast_into_vec(); // Convert back into vec
        for (&a, &b) in vecf64.iter().zip(nu_vec.iter()) {
            assert!((a - b).abs() < 1e-6f64 * f64::max(a, b).abs());
        }

        let vecf64 = vec![1f64, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        let buf = VecUnit::from(vecf64.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.cast_into_vec(); // Convert back into vec
        let vecf32 = vec![1f32, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        assert_eq!(vecf32, nu_vec);
        let buf = VecUnit::from(vecf32.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.cast_into_vec(); // Convert back into vec
        for (&a, &b) in vecf64.iter().zip(nu_vec.iter()) {
            assert!((a - b).abs() < 1e-6 * f64::max(a, b).abs());
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Foo {
        a: u8,
        b: i64,
        c: f32,
    }

    #[test]
    fn from_empty_vec_test() {
        let vec: Vec<u32> = Vec::new();
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<u32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Foo> = Vec::new();
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Foo> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn from_struct_test() {
        let f1 = Foo {
            a: 3,
            b: -32,
            c: 54.2,
        };
        let f2 = Foo {
            a: 33,
            b: -3342432412,
            c: 323454.2,
        };
        let vec = vec![f1.clone(), f2.clone()];
        let buf = VecUnit::from(vec.clone()); // Convert into buffer
        assert_eq!(f1, buf.get_ref_as::<Foo>(0).unwrap().clone());
        assert_eq!(f2, buf.get_ref_as::<Foo>(1).unwrap().clone());
        let nu_vec: Vec<Foo> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn iter_test() {
        // Check iterating over data with a larger size than 8 bits.
        let vec_f32 = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let buf = VecUnit::from(vec_f32.clone()); // Convert into buffer
        for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        // Check iterating over data with the same size.
        let vec_u8 = vec![1u8, 3, 4, 1, 2, 4, 128, 32];
        let buf = VecUnit::from(vec_u8.clone()); // Convert into buffer
        for (i, &val) in buf.iter_as::<u8>().unwrap().enumerate() {
            assert_eq!(val, vec_u8[i]);
        }
    }

    #[test]
    fn large_sizes_test() {
        for i in 1000000..1000010 {
            let vec = vec![32u8; i];
            let buf = VecUnit::from(vec.clone()); // Convert into buffer
            let nu_vec: Vec<u8> = buf.into_vec().unwrap(); // Convert back into vec
            assert_eq!(vec, nu_vec);
        }
    }

    /// This test checks that an error is returned whenever the user tries to access data with the
    /// wrong type data.
    #[test]
    fn wrong_type_test() {
        let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let mut buf = VecUnit::from(vec.clone()); // Convert into buffer
        assert_eq!(vec, buf.copy_into_vec::<f32>().unwrap());

        assert!(buf.copy_into_vec::<f64>().is_none());
        assert!(buf.as_slice_as::<f64>().is_none());
        assert!(buf.as_mut_slice_as::<u8>().is_none());
        assert!(buf.iter_as::<[f32; 3]>().is_none());
        assert!(buf.get_as::<i32>(0).is_none());
        assert!(buf.get_ref_as::<i32>(1).is_none());
        assert!(buf.get_mut_as::<i32>(2).is_none());
    }

    /// Test pushing values and bytes to a buffer.
    #[test]
    fn push_test() {
        let mut vec_f32 = vec![1.0_f32, 23.0, 0.01];
        let mut buf = VecUnit::from(vec_f32.clone()); // Convert into buffer
        for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        vec_f32.push(42.0f32);
        buf.push_as(42.0f32).unwrap(); // must provide explicit type

        for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        vec_f32.push(11.43);
        buf.push_as(11.43f32).unwrap();

        for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }
    }

    /// Test iterating over chunks of data without having to interpret them.
    #[test]
    fn vecvoid_byte_chunks_test() {
        let vec_f32 = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let buf = VecVoid::from_vec(vec_f32.clone()); // Convert into VecVoid

        for (i, val) in buf.byte_chunks().enumerate() {
            assert_eq!(val.len(), 4);
            // SAFETY: This is safe since the underlying bytes were initialized.
            // This also will not cause issues on different platforms since
            // conversion to and from bytes is happening on the same platform.
            unsafe {
                assert_eq!(*(val.as_ptr() as *const f32), vec_f32[i]);
            }
        }
    }

    /// Check the byte getters on `VecVoid`.
    #[test]
    fn vecvoid_byte_getters() {
        let vec_u8 = vec![1, 2, 3, 4];
        let mut buf = VecVoid::from_vec(vec_u8.clone()); // Convert into VecVoid

        // Test byte getters
        // SAFETY: since all bytes have been initialized, this is safe.
        for i in 0..4 {
            assert_eq!(
                unsafe { std::mem::transmute::<_, &[u8]>(buf.get_bytes(i)) },
                &[vec_u8[i]][..]
            );
            assert_eq!(
                unsafe { std::mem::transmute::<_, &mut [u8]>(buf.get_bytes_mut(i)) },
                &[vec_u8[i]][..]
            );
        }
    }

    /// Test appending to a data buffer from another data buffer.
    #[test]
    fn append_test() {
        let mut buf = VecUnit::with_type::<f32>(); // Create an empty buffer.

        let data = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        // Append an ordianry vector of data.
        let mut other_buf = VecUnit::from_vec(data.clone());
        buf.append(&mut other_buf);

        assert!(other_buf.is_empty());

        for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
            assert_eq!(val, data[i]);
        }
    }

    /// Test that `VecCopy` can be extended by or constructed from an iterator.
    #[test]
    fn extend_and_collect() {
        let mut buf = VecUnit::with_type::<f32>();
        let other = VecUnit::from(vec![1.0_f32, 23.0, 0.01]);
        buf.extend(other.iter());

        let buf2: VecUnit = other.iter().collect();

        assert_eq!(buf.as_slice_as::<f32>(), buf2.as_slice_as::<f32>());
    }

    /// Test dynamically sized vtables.
    #[cfg(feature = "traits")]
    #[test]
    fn dynamic_vtables() {
        use crate::into_dyn;
        let buf = VecUnit::with_type::<u8>(); // Create an empty buffer.

        let mut buf_dyn = into_dyn![VecCopy<dyn Any>](buf);

        buf_dyn.push(CopyValueRef::<()>::new(&1u8));
        buf_dyn.push(CopyValueRef::<()>::new(&100u8));
        buf_dyn.push(CopyValueRef::<()>::new(&23u8));

        let vec: Vec<u8> = buf_dyn.into_vec().unwrap();

        assert_eq!(vec, vec![1u8, 100, 23]);
    }
}
