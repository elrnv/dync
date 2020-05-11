use std::{
    any::{Any, TypeId},
    mem::size_of,
    slice,
};

use crate::index_slice::*;
use crate::value::*;
use crate::CopyElem;
use crate::{ElementBytes, ElementBytesMut};

#[cfg(not(feature = "shared-vtables"))]
use std::boxed::Box as Ptr;
#[cfg(feature = "shared-vtables")]
use std::rc::Rc as Ptr;

/*
 * Immutable slice
 */

#[derive(Clone)]
pub struct SliceCopy<'a, V = ()>
where
    V: ?Sized,
{
    /// Raw data stored as bytes.
    pub(crate) data: &'a [u8],
    /// Number of bytes occupied by an element of this buffer.
    ///
    /// Note: We store this instead of length because it gives us the ability to get the type size
    /// when the buffer is empty.
    pub(crate) element_size: usize,
    /// Type encoding for hiding the type of data from the compiler.
    pub(crate) element_type_id: TypeId,

    pub(crate) vtable: VTableRef<'a, V>,
}

impl<'a, V> SliceCopy<'a, V> {
    /// Construct a `SliceCopy` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: CopyElem>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        // This is safe since `CopyElem` is `Copy`.
        unsafe { Self::from_slice_non_copy(slice) }
    }

    /// It is unsafe to call this for `T` that is not `Copy`.
    #[inline]
    pub(crate) unsafe fn from_slice_non_copy<T: Any>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        let element_size = size_of::<T>();
        SliceCopy {
            data: std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * element_size,
            ),
            element_size,
            element_type_id: TypeId::of::<T>(),
            vtable: VTableRef::Box(Box::new(V::build_vtable())),
        }
    }
}

impl<'a, V> From<SliceCopy<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceCopy<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable,
        }
    }
}

impl<'a, V: Clone> From<SliceCopy<'a, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(slice: SliceCopy<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable.into_owned(),
        }
    }
}

impl<'a, V: ?Sized> SliceCopy<'a, V> {
    /// Convert this collection into its raw components.
    ///
    /// This function exists mainly to enable the `into_dyn` macro until `CoerceUnsized` is
    /// stabilized.
    #[inline]
    pub unsafe fn into_raw_parts(self) -> (&'a [u8], usize, TypeId, VTableRef<'a, V>) {
        let SliceCopy {
            data,
            element_size,
            element_type_id,
            vtable,
        } = self;
        (data, element_size, element_type_id, vtable)
    }

    /// This is very unsafe to use.
    ///
    /// Almost exclusively the only inputs that work here are the ones returned by
    /// `into_raw_parts`.
    ///
    /// This function should not be used other than in internal APIs. It exists to enable the
    /// `into_dyn` macro until `CoerceUsize` is stabilized.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: &'a [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: impl Into<VTableRef<'a, V>>,
    ) -> Self {
        SliceCopy {
            data,
            element_size,
            element_type_id,
            vtable: vtable.into(),
        }
    }

    /// Upcast the `SliceCopy` into a more general base `SliceCopy`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> SliceCopy<'a, U>
    where
        V: Clone,
    {
        self.upcast_with(U::from)
    }

    // Helper for upcast implementations.
    #[inline]
    pub(crate) fn upcast_with<U>(self, f: impl FnOnce(V) -> U) -> SliceCopy<'a, U>
    where
        V: Clone,
    {
        SliceCopy {
            data: self.data,
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Box(Box::new(f(self.vtable.take()))),
        }
    }

    /// Check if the current buffer contains elements of the specified type.
    ///
    /// Returns `Some(self)` if the type matches and `None` otherwise.
    #[inline]
    pub fn check<T: Any>(&self) -> Option<&Self> {
        if TypeId::of::<T>() != self.element_type_id() {
            None
        } else {
            Some(self)
        }
    }

    /// Construct a clone of the current slice with a reduced lifetime.
    ///
    /// This is equivalent to calling `subslice` with the entire range.
    #[inline]
    pub fn reborrow(&self) -> SliceCopy<V> {
        SliceCopy {
            data: self.data,
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    /*
     * Accessors
     */

    /// Get the `TypeId` of data stored within this buffer.
    #[inline]
    pub fn element_type_id(&self) -> TypeId {
        self.element_type_id
    }

    /// Get the number of elements stored in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.data.len() % self.element_size, 0);
        self.data.len() / self.element_size // element_size is guaranteed to be strictly positive
    }

    /// Check if there are any elements stored in this buffer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the size of the element type in bytes.
    #[inline]
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Return an iterator to a slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    ///
    /// # Examples
    /// ```
    /// use dync::SliceCopy;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf = SliceCopy::<()>::from_slice(vec.as_slice());
    /// for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
    ///     assert_eq!(val, vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&self) -> Option<slice::Iter<T>> {
        self.as_slice::<T>().map(|x| x.iter())
    }

    /// Append copied items from this buffer to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise.
    #[inline]
    pub fn append_copy_to_vec<'b, T: CopyElem>(
        &self,
        vec: &'b mut Vec<T>,
    ) -> Option<&'b mut Vec<T>> {
        let iter = self.iter_as()?;
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

    /// Borrow this slice as a typed slice.
    ///
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Option<&[T]> {
        let ptr = self.check::<T>()?.data.as_ptr() as *const T;
        Some(unsafe { slice::from_raw_parts(ptr, self.len()) })
    }

    /// Get `i`'th element of the buffer.
    #[inline]
    pub fn get_as<T: CopyElem>(&self, i: usize) -> Option<&T> {
        assert!(i < self.len());
        let ptr = self.check::<T>()?.data.as_ptr() as *const T;
        Some(unsafe { &*ptr.add(i) })
    }

    /*
     * Value API. This allows users to manipulate contained data without knowing the element type.
     */

    /// Return an iterator over untyped value references stored in this buffer.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    ///
    /// # Examples
    /// ```
    /// use dync::SliceCopy;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf: SliceCopy = SliceCopy::from(vec.as_slice());
    /// for (i, val) in buf.iter().enumerate() {
    ///     assert_eq!(val.downcast::<f32>().unwrap(), &vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = CopyValueRef<V>> {
        self.byte_chunks().map(move |bytes| unsafe {
            CopyValueRef::from_raw_parts(bytes, self.element_type_id(), self.vtable.as_ref())
        })
    }

    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = SliceCopy<V>> {
        let &SliceCopy {
            ref data,
            element_size,
            element_type_id,
            ref vtable,
        } = self;
        data.chunks_exact(element_size * chunk_size)
            .map(move |data| unsafe {
                SliceCopy::from_raw_parts(data, element_size, element_type_id, vtable.as_ref())
            })
    }

    pub fn split_at(&self, mid: usize) -> (SliceCopy<V>, SliceCopy<V>) {
        let &SliceCopy {
            ref data,
            element_size,
            element_type_id,
            ref vtable,
        } = self;
        unsafe {
            let (l, r) = data.split_at(mid * element_size);
            (
                SliceCopy::from_raw_parts(l, element_size, element_type_id, vtable.as_ref()),
                SliceCopy::from_raw_parts(r, element_size, element_type_id, vtable.as_ref()),
            )
        }
    }

    /// Get a immutable value reference to the element at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> CopyValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            CopyValueRef::from_raw_parts(
                &self.index_byte_slice(i),
                self.element_type_id(),
                self.vtable.as_ref(),
            )
        }
    }

    /// Get an immutable subslice representing the given range of indices.
    #[inline]
    pub fn subslice<I>(&self, i: I) -> SliceCopy<V>
    where
        I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
    {
        SliceCopy {
            data: &self.data[i.scale_range(self.element_size())],
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    /// Conver this slice into a mutable subslice representing the given range of indices.
    #[inline]
    pub fn into_subslice<I>(self, i: I) -> SliceCopy<'a, V>
    where
        I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
    {
        let element_size = self.element_size();
        SliceCopy {
            data: &self.data[i.scale_range(element_size)],
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: self.vtable,
        }
    }

    /*
     * Advanced functions
     */

    /// Iterate over element sized chunks of bytes.
    #[inline]
    pub(crate) fn byte_chunks(&self) -> impl Iterator<Item = &[u8]> {
        self.bytes().chunks_exact(self.element_size())
    }
}

impl<'a, V: ?Sized> ElementBytes for SliceCopy<'a, V> {
    #[inline]
    fn element_size(&self) -> usize {
        self.element_size
    }
    #[inline]
    fn bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Convert a `&[T]` to a `SliceCopy`.
impl<'a, T, V> From<&'a [T]> for SliceCopy<'a, V>
where
    T: CopyElem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a [T]) -> SliceCopy<'a, V> {
        SliceCopy::from_slice(s)
    }
}

/*
 * Mutable Slice
 */

pub struct SliceCopyMut<'a, V = ()>
where
    V: ?Sized,
{
    /// Raw data stored as bytes.
    pub(crate) data: &'a mut [u8],
    /// Number of bytes occupied by an element of this buffer.
    ///
    /// Note: We store this instead of length because it gives us the ability to get the type size
    /// when the buffer is empty.
    pub(crate) element_size: usize,
    /// Type encoding for hiding the type of data from the compiler.
    pub(crate) element_type_id: TypeId,
    pub(crate) vtable: VTableRef<'a, V>,
}

impl<'a, V> SliceCopyMut<'a, V> {
    /// Construct a `SliceCopyMut` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: CopyElem>(slice: &'a mut [T]) -> SliceCopyMut<'a, V>
    where
        V: VTable<T>,
    {
        // This is safe since `CopyElem` is `Copy`.
        unsafe { Self::from_slice_non_copy(slice) }
    }

    /// It is unsafe to call this for `T` that is not `Copy`.
    #[inline]
    pub(crate) unsafe fn from_slice_non_copy<T: Any>(slice: &'a mut [T]) -> SliceCopyMut<'a, V>
    where
        V: VTable<T>,
    {
        let element_size = size_of::<T>();
        SliceCopyMut {
            data: std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut u8,
                slice.len() * element_size,
            ),
            element_size,
            element_type_id: TypeId::of::<T>(),
            vtable: VTableRef::Box(Box::new(V::build_vtable())),
        }
    }
}

impl<'a, V: ?Sized> SliceCopyMut<'a, V> {
    /// Convert this collection into its raw components.
    ///
    /// This function exists mainly to enable the `into_dyn` macro until `CoerceUnsized` is
    /// stabilized.
    #[inline]
    pub unsafe fn into_raw_parts(self) -> (&'a mut [u8], usize, TypeId, VTableRef<'a, V>) {
        let SliceCopyMut {
            data,
            element_size,
            element_type_id,
            vtable,
        } = self;
        (data, element_size, element_type_id, vtable)
    }

    /// This is very unsafe to use.
    ///
    /// Almost exclusively the only inputs that work here are the ones returned by
    /// `into_raw_parts`.
    ///
    /// This function should not be used other than in internal APIs. It exists to enable the
    /// `into_dyn` macro until `CoerceUsize` is stabilized.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: &'a mut [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: impl Into<VTableRef<'a, V>>,
    ) -> SliceCopyMut<'a, V> {
        SliceCopyMut {
            data,
            element_size,
            element_type_id,
            vtable: vtable.into(),
        }
    }

    /// Swap the values at the two given indices.
    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        ElementBytesMut::swap(self, i, j);
    }

    /// Upcast the `SliceCopyMut` into a more general base `SliceCopy`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> SliceCopyMut<'a, U>
    where
        V: Clone,
    {
        self.upcast_with(U::from)
    }

    // Helper for upcast implementations.
    #[inline]
    pub(crate) fn upcast_with<U>(self, f: impl FnOnce(V) -> U) -> SliceCopyMut<'a, U>
    where
        V: Clone,
    {
        SliceCopyMut {
            data: self.data,
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Box(Box::new(f(self.vtable.take()))),
        }
    }

    /// Copy data from a given slice into the current slice.
    ///
    /// # Panics
    ///
    /// This function will panic if the given slice has as different size than current.
    #[inline]
    pub fn copy_from_slice<T: CopyElem>(&mut self, slice: &[T]) -> Option<&mut Self> {
        let this_slice = self.as_slice::<T>()?;
        this_slice.copy_from_slice(slice);
        Some(self)
    }

    /// Check if the current buffer contains elements of the specified type. Returns `Some(self)`
    /// if the type matches and `None` otherwise.
    #[inline]
    pub fn check<T: Any>(&mut self) -> Option<&mut Self> {
        if TypeId::of::<T>() != self.element_type_id() {
            None
        } else {
            Some(self)
        }
    }

    /// Construct a clone of the current slice with a reduced lifetime.
    ///
    /// This is equivalent to calling `subslice` with the entire range.
    #[inline]
    pub fn reborrow(&self) -> SliceCopy<V> {
        SliceCopy {
            data: self.data,
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    /// Construct a clone of the current slice with a reduced lifetime.
    ///
    /// This is equivalent to calling `subslice_mut` with the entire range.
    #[inline]
    pub fn reborrow_mut(&mut self) -> SliceCopyMut<V> {
        SliceCopyMut {
            data: self.data,
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    /*
     * Accessors
     */

    /// Get the `TypeId` of data stored within this buffer.
    #[inline]
    pub fn element_type_id(&self) -> TypeId {
        self.element_type_id
    }

    /// Get the number of elements stored in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.data.len() % self.element_size, 0);
        self.data.len() / self.element_size // element_size is guaranteed to be strictly positive
    }

    /// Check if there are any elements stored in this buffer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return an iterator to a slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    ///
    /// # Examples
    /// ```
    /// use dync::SliceCopyMut;
    /// let mut vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let mut buf = SliceCopyMut::<()>::from(vec.as_mut_slice());
    /// for val in buf.iter_as::<f32>().unwrap() {
    ///     *val += 1.0_f32;
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&mut self) -> Option<slice::IterMut<T>> {
        self.as_slice::<T>().map(|x| x.iter_mut())
    }

    /// Append copied items from this buffer to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise.
    #[inline]
    pub fn append_copy_to_vec<'b, T: CopyElem>(
        &self,
        vec: &'b mut Vec<T>,
    ) -> Option<&'b mut Vec<T>> {
        let slice = SliceCopy::from(self);
        vec.extend(slice.iter_as()?);
        Some(vec)
    }

    /// Copies contents of `self` into the given `Vec`.
    #[inline]
    pub fn copy_into_vec<T: CopyElem>(self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        match self.append_copy_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }

    /// Convert this `SliceCopyMut` into a typed slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice<T: Any>(&mut self) -> Option<&mut [T]> {
        let len = self.len();
        let ptr = self.check::<T>()?.data.as_ptr() as *mut T;
        Some(unsafe { slice::from_raw_parts_mut(ptr, len) })
    }

    /// Get `i`'th element of the buffer.
    #[inline]
    pub fn get_as<T: CopyElem>(&mut self, i: usize) -> Option<&mut T> {
        assert!(i < self.len());
        let ptr = self.check::<T>()?.data.as_mut_ptr() as *mut T;
        Some(unsafe { &mut *ptr.add(i) })
    }

    /// Rotates the slice in-place such that the first `mid` elements of the slice move to the end
    /// while the last `self.len() - mid` elements move to the front. After calling `rotate_left`,
    /// the element previously at index `mid` will become the first element in the slice.
    ///
    /// # Example
    ///
    /// ```
    /// use dync::*;
    /// let mut vec = vec![1u32,2,3,4,5];
    /// let mut buf = SliceCopyMut::<()>::from_slice(vec.as_mut_slice());
    /// buf.rotate_left(3);
    /// assert_eq!(buf.as_slice::<u32>().unwrap(), &[4,5,1,2,3]);
    /// ```
    #[inline]
    pub fn rotate_left(&mut self, mid: usize) {
        self.data.rotate_left(mid * self.element_size);
    }

    /// Rotates the slice in-place such that the first `self.len() - k` elements of the slice move
    /// to the end while the last `k` elements move to the front. After calling `rotate_right`, the
    /// element previously at index `k` will become the first element in the slice.
    ///
    /// # Example
    ///
    /// ```
    /// use dync::*;
    /// let mut vec = vec![1u32,2,3,4,5];
    /// let mut buf = SliceCopyMut::<()>::from_slice(vec.as_mut_slice());
    /// buf.rotate_right(3);
    /// assert_eq!(buf.as_slice::<u32>().unwrap(), &[3,4,5,1,2]);
    /// ```
    #[inline]
    pub fn rotate_right(&mut self, k: usize) {
        self.data.rotate_right(k * self.element_size);
    }

    /*
     * Value API. This allows users to manipulate contained data without knowing the element type.
     */

    /// Return an iterator over untyped value references stored in this slice.
    ///
    /// In contrast to `iter_as`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    #[inline]
    pub fn iter(&mut self) -> impl Iterator<Item = CopyValueMut<V>> {
        let &mut SliceCopyMut {
            ref mut data,
            element_size,
            element_type_id,
            ref vtable,
        } = self;
        data.chunks_exact_mut(element_size)
            .map(move |bytes| unsafe {
                CopyValueMut::from_raw_parts(bytes, element_type_id, vtable.as_ref())
            })
    }

    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = SliceCopy<V>> {
        let &SliceCopyMut {
            ref data,
            element_size,
            element_type_id,
            ref vtable,
        } = self;
        data.chunks_exact(element_size * chunk_size)
            .map(move |data| unsafe {
                SliceCopy::from_raw_parts(data, element_size, element_type_id, vtable.as_ref())
            })
    }

    #[inline]
    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> impl Iterator<Item = SliceCopyMut<V>> {
        let &mut SliceCopyMut {
            ref mut data,
            element_size,
            element_type_id,
            ref vtable,
        } = self;
        data.chunks_exact_mut(element_size * chunk_size)
            .map(move |data| unsafe {
                SliceCopyMut::from_raw_parts(data, element_size, element_type_id, vtable.as_ref())
            })
    }

    pub fn split_at(&mut self, mid: usize) -> (SliceCopyMut<V>, SliceCopyMut<V>) {
        let &mut SliceCopyMut {
            ref mut data,
            element_size,
            element_type_id,
            ref vtable,
        } = self;
        unsafe {
            let (l, r) = data.split_at_mut(mid * element_size);
            (
                SliceCopyMut::from_raw_parts(l, element_size, element_type_id, vtable.as_ref()),
                SliceCopyMut::from_raw_parts(r, element_size, element_type_id, vtable.as_ref()),
            )
        }
    }

    /// Get a immutable value reference to the element at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> CopyValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            CopyValueRef::from_raw_parts(
                &self.index_byte_slice(i),
                self.element_type_id(),
                self.vtable.as_ref(),
            )
        }
    }

    /// Get a mutable value reference to the element at index `i`.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> CopyValueMut<V> {
        debug_assert!(i < self.len());
        let element_type_id = self.element_type_id();
        let element_bytes = self.index_byte_range(i);

        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            CopyValueMut::from_raw_parts(
                &mut self.data[element_bytes],
                element_type_id,
                self.vtable.as_ref(),
            )
        }
    }

    /// Get an immutable subslice representing the given range of indices.
    #[inline]
    pub fn subslice<I>(&self, i: I) -> SliceCopy<V>
    where
        I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
    {
        SliceCopy {
            data: &self.data[i.scale_range(self.element_size())],
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    /// Get a mutable subslice representing the given range of indices.
    #[inline]
    pub fn subslice_mut<I>(&mut self, i: I) -> SliceCopyMut<V>
    where
        I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
    {
        let element_size = self.element_size();
        SliceCopyMut {
            data: &mut self.data[i.scale_range(element_size)],
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: VTableRef::Ref(self.vtable.as_ref()),
        }
    }

    /// Conver this slice into a mutable subslice representing the given range of indices.
    #[inline]
    pub fn into_subslice<I>(self, i: I) -> SliceCopyMut<'a, V>
    where
        I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
    {
        let element_size = self.element_size();
        SliceCopyMut {
            data: &mut self.data[i.scale_range(element_size)],
            element_size: self.element_size,
            element_type_id: self.element_type_id,
            vtable: self.vtable,
        }
    }
}

impl<'a, V> From<SliceCopyMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceCopyMut<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable,
        }
    }
}

impl<'a, V: Clone> From<SliceCopyMut<'a, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(slice: SliceCopyMut<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable.into_owned(),
        }
    }
}

impl<'a, V: ?Sized> ElementBytes for SliceCopyMut<'a, V> {
    #[inline]
    fn element_size(&self) -> usize {
        self.element_size
    }
    #[inline]
    fn bytes(&self) -> &[u8] {
        &self.data
    }
}

impl<'a, V: ?Sized> ElementBytesMut for SliceCopyMut<'a, V> {
    #[inline]
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Convert a `&mut [T]` to a `SliceCopyMut`.
impl<'a, T, V> From<&'a mut [T]> for SliceCopyMut<'a, V>
where
    T: CopyElem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a mut [T]) -> SliceCopyMut<'a, V> {
        SliceCopyMut::from_slice(s)
    }
}

impl<'a, V: ?Sized> From<SliceCopyMut<'a, V>> for SliceCopy<'a, V> {
    #[inline]
    fn from(s: SliceCopyMut<'a, V>) -> SliceCopy<'a, V> {
        SliceCopy {
            data: s.data,
            element_size: s.element_size,
            element_type_id: s.element_type_id,
            vtable: s.vtable,
        }
    }
}

impl<'b, 'a: 'b, V: ?Sized> From<&'b SliceCopyMut<'a, V>> for SliceCopy<'b, V> {
    #[inline]
    fn from(s: &'b SliceCopyMut<'a, V>) -> SliceCopy<'b, V> {
        unsafe {
            SliceCopy::from_raw_parts(s.data, s.element_size, s.element_type_id, s.vtable.as_ref())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VecCopy;

    /// Test dynamically sized vtables.
    #[test]
    fn dynamic_vtables() {
        use crate::into_dyn;
        let vec = vec![1u8, 100, 23];

        // SliceCopyMut
        let mut buf = VecCopy::<()>::from_vec(vec.clone());
        let slice = buf.as_mut_slice();
        let mut slice_dyn = into_dyn![SliceCopyMut<dyn Any>](slice);
        *slice_dyn.get_mut(1).downcast::<u8>().unwrap() += 1u8;
        assert_eq!(buf.into_vec::<u8>().unwrap(), vec![1u8, 101, 23]);

        // SliceCopy
        let buf = VecCopy::<()>::from_vec(vec.clone());
        let slice = buf.as_slice();
        let slice_dyn = into_dyn![SliceCopy<dyn Any>](slice);
        assert_eq!(*slice_dyn.get(0).downcast::<u8>().unwrap(), 1u8);
        assert_eq!(*slice_dyn.get(1).downcast::<u8>().unwrap(), 100u8);
        assert_eq!(*slice_dyn.get(2).downcast::<u8>().unwrap(), 23u8);
    }
}
