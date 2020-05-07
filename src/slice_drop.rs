use std::{
    any::{Any, TypeId},
    slice,
};

use crate::index_slice::*;
use crate::slice_copy::*;
use crate::traits::*;
use crate::value::*;
use crate::Elem;
use crate::ElementBytes;

/*
 * Immutable slice
 */

#[derive(Clone)]
pub struct SliceDrop<'a, V> {
    pub(crate) data: SliceCopy<'a, (DropFn, V)>,
}

impl<'a, V> SliceDrop<'a, V> {
    pub(crate) unsafe fn from_raw_parts(
        data: &'a [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: impl Into<VTableRef<'a, (DropFn, V)>>,
    ) -> Self {
        SliceDrop {
            data: SliceCopy::from_raw_parts(data, element_size, element_type_id, vtable),
        }
    }

    /// Construct a `SliceDrop` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        SliceDrop {
            data: unsafe { SliceCopy::from_slice_non_copy(slice) },
        }
    }

    /// Upcast the `SliceDrop` into a more general base `SliceDrop`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> SliceDrop<'a, U>
    where
        V: Clone,
    {
        SliceDrop {
            data: self.data.upcast_with(|v: (DropFn, V)| (v.0, U::from(v.1))),
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

    /// Get the size of the element type in bytes.
    #[inline]
    pub fn element_size(&self) -> usize {
        self.data.element_size()
    }

    /// Return an iterator to a slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    ///
    /// # Examples
    /// ```
    /// use dync::SliceDrop;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf = SliceDrop::<()>::from_slice(vec.as_slice());
    /// for (i, &val) in buf.iter_as::<f32>().unwrap().enumerate() {
    ///     assert_eq!(val, vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&self) -> Option<slice::Iter<T>> {
        self.as_slice::<T>().map(|x| x.iter())
    }

    /// Borrow this slice as a typed slice.
    ///
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Option<&[T]> {
        let ptr = self.check::<T>()?.data.data.as_ptr() as *const T;
        Some(unsafe { slice::from_raw_parts(ptr, self.len()) })
    }

    /// Get `i`'th element of the buffer.
    #[inline]
    pub fn get_as<T: Elem>(&self, i: usize) -> Option<&T> {
        assert!(i < self.len());
        let ptr = self.check::<T>()?.data.data.as_ptr() as *const T;
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
    /// use dync::SliceDrop;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf = SliceDrop::<()>::from(vec.as_slice());
    /// for (i, val) in buf.iter().enumerate() {
    ///     assert_eq!(val.downcast::<f32>().unwrap(), &vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ValueRef<V>> {
        self.data.byte_chunks().map(move |bytes| unsafe {
            ValueRef::from_raw_parts(bytes, self.element_type_id(), self.data.vtable.as_ref())
        })
    }

    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = SliceDrop<V>> {
        self.data
            .chunks_exact(chunk_size)
            .map(|data| SliceDrop { data })
    }

    #[inline]
    pub fn split_at(&self, mid: usize) -> (SliceDrop<V>, SliceDrop<V>) {
        let (l, r) = self.data.split_at(mid);
        (SliceDrop { data: l }, SliceDrop { data: r })
    }

    /*
     * Advanced methods to probe buffer internals.
     */

    /// Get a `const` reference to the byte slice of the `i`'th element of the buffer.
    #[inline]
    pub(crate) fn index_byte_slice(&self, i: usize) -> &[u8] {
        self.data.index_byte_slice(i)
    }
}

impl<'a, V: HasClone> SliceDrop<'a, V> {
    /// Append cloned items from this buffer to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise.
    #[inline]
    pub fn append_clone_to_vec<'b, T: Elem + Clone>(
        &self,
        vec: &'b mut Vec<T>,
    ) -> Option<&'b mut Vec<T>> {
        let iter = self.iter_as()?;
        vec.extend(iter.cloned());
        Some(vec)
    }

    /// Clones contents of `self` into the given `Vec`.
    #[inline]
    pub fn clone_into_vec<T: Elem + Clone>(&self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        match self.append_clone_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }
}

impl<'a, V> IndexSlice<'a, usize> for SliceDrop<'a, V> {
    type Output = ValueRef<'a, V>;

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    fn get(&'a self, i: usize) -> ValueRef<'a, V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueRef::from_raw_parts(
                self.index_byte_slice(i),
                self.element_type_id(),
                self.data.vtable.as_ref(),
            )
        }
    }
}

impl<'a, V, I> IndexSlice<'a, I> for SliceDrop<'a, V>
where
    I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
{
    type Output = SliceDrop<'a, V>;
    fn get(&'a self, i: I) -> Self::Output {
        SliceDrop {
            data: self.data.get(i),
        }
    }
}

/// Convert a `&[T]` to a `SliceDrop`.
impl<'a, T, V> From<&'a [T]> for SliceDrop<'a, V>
where
    T: Elem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a [T]) -> SliceDrop<'a, V> {
        SliceDrop::from_slice(s)
    }
}

/*
 * Mutable Slice
 */

pub struct SliceDropMut<'a, V> {
    pub(crate) data: SliceCopyMut<'a, (DropFn, V)>,
}

impl<'a, V> SliceDropMut<'a, V> {
    pub(crate) unsafe fn from_raw_parts(
        data: &'a mut [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: impl Into<VTableRef<'a, (DropFn, V)>>,
    ) -> SliceDropMut<'a, V> {
        SliceDropMut {
            data: SliceCopyMut::from_raw_parts(data, element_size, element_type_id, vtable),
        }
    }

    /// Construct a `SliceDropMut` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &'a mut [T]) -> SliceDropMut<'a, V>
    where
        V: VTable<T>,
    {
        SliceDropMut {
            data: unsafe { SliceCopyMut::from_slice_non_copy(slice) },
        }
    }

    /// Upcast the `SliceDropMut` into a more general base `SliceDropMut`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> SliceDropMut<'a, U>
    where
        V: Clone,
    {
        SliceDropMut {
            data: self.data.upcast_with(|v: (DropFn, V)| (v.0, U::from(v.1))),
        }
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

    /// Swap the values at the two given indices.
    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        // We don't need to worry about drops or clones here.
        self.data.swap(i, j);
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

    /// Get the size of the element type in bytes.
    #[inline]
    pub fn element_size(&self) -> usize {
        self.data.element_size()
    }

    /// Return an iterator to a slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    ///
    /// # Examples
    /// ```
    /// use dync::SliceDropMut;
    /// let mut vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let mut buf = SliceDropMut::<()>::from(vec.as_mut_slice());
    /// for val in buf.iter_as::<f32>().unwrap() {
    ///     *val += 1.0_f32;
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&mut self) -> Option<slice::IterMut<T>> {
        self.as_slice::<T>().map(|x| x.iter_mut())
    }

    /// Convert this `SliceDropMut` into a typed slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice<T: Any>(&mut self) -> Option<&mut [T]> {
        let len = self.len();
        let ptr = self.check::<T>()?.data.data.as_ptr() as *mut T;
        Some(unsafe { slice::from_raw_parts_mut(ptr, len) })
    }

    /// Get `i`'th element of the buffer.
    #[inline]
    pub fn get_as<T: Elem>(&mut self, i: usize) -> Option<&mut T> {
        assert!(i < self.len());
        let ptr = self.check::<T>()?.data.data.as_mut_ptr() as *mut T;
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
    /// let mut buf = SliceDropMut::<()>::from_slice(vec.as_mut_slice());
    /// buf.rotate_left(3);
    /// assert_eq!(buf.as_slice::<u32>().unwrap(), &[4,5,1,2,3]);
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
    /// let mut vec = vec![1u32,2,3,4,5];
    /// let mut buf = SliceDropMut::<()>::from_slice(vec.as_mut_slice());
    /// buf.rotate_right(3);
    /// assert_eq!(buf.as_slice::<u32>().unwrap(), &[3,4,5,1,2]);
    /// ```
    #[inline]
    pub fn rotate_right(&mut self, k: usize) {
        self.data.rotate_right(k);
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
    pub fn iter(&mut self) -> impl Iterator<Item = ValueMut<V>>
    where
        V: Clone,
    {
        let SliceCopyMut {
            ref mut data,
            element_size,
            element_type_id,
            ref vtable,
        } = self.data;
        data.chunks_exact_mut(element_size)
            .map(move |bytes| unsafe {
                ValueMut::from_raw_parts(bytes, element_type_id, vtable.as_ref())
            })
    }

    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = SliceDrop<V>> {
        self.data
            .chunks_exact(chunk_size)
            .map(|data| SliceDrop { data })
    }

    #[inline]
    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> impl Iterator<Item = SliceDropMut<V>> {
        self.data
            .chunks_exact_mut(chunk_size)
            .map(|data| SliceDropMut { data })
    }

    #[inline]
    pub fn split_at(&mut self, mid: usize) -> (SliceDropMut<V>, SliceDropMut<V>) {
        let (l, r) = self.data.split_at(mid);
        (SliceDropMut { data: l }, SliceDropMut { data: r })
    }
}

impl<'a, V: HasClone> SliceDropMut<'a, V> {
    /// Clone data from a given slice into the current slice.
    ///
    /// # Panics
    ///
    /// This function will panic if the given slice has as different size than current.
    #[inline]
    pub fn clone_from_slice<T: Elem + Clone>(&mut self, slice: &[T]) -> Option<&mut Self> {
        let this_slice = self.as_slice::<T>()?;
        this_slice.clone_from_slice(slice);
        Some(self)
    }

    /// Append cloned items from this slice to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise.
    #[inline]
    pub fn append_clone_to_vec<'b, T: Elem + Clone>(
        &self,
        vec: &'b mut Vec<T>,
    ) -> Option<&'b mut Vec<T>> {
        SliceDrop::from(self).append_clone_to_vec(vec)?;
        Some(vec)
    }

    /// Clones contents of `self` into the given `Vec`.
    #[inline]
    pub fn clone_into_vec<T: Elem + Clone>(self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        match self.append_clone_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }
}

impl<'a, V> IndexSlice<'a, usize> for SliceDropMut<'a, V> {
    type Output = ValueRef<'a, V>;

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    fn get(&'a self, i: usize) -> ValueRef<'a, V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueRef::from_raw_parts(
                self.data.index_byte_slice(i),
                self.element_type_id(),
                self.data.vtable.as_ref(),
            )
        }
    }
}

impl<'a, V> IndexSliceMut<'a, usize> for SliceDropMut<'a, V> {
    type Output = ValueMut<'a, V>;

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    fn get_mut(&'a mut self, i: usize) -> ValueMut<'a, V> {
        let CopyValueMut {
            bytes,
            type_id,
            vtable,
        } = self.data.get_mut(i);

        ValueMut {
            bytes,
            type_id,
            vtable,
        }
    }
}

impl<'a, V, I> IndexSlice<'a, I> for SliceDropMut<'a, V>
where
    I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
{
    type Output = SliceDrop<'a, V>;
    fn get(&'a self, i: I) -> Self::Output {
        SliceDrop {
            data: self.data.get(i),
        }
    }
}

impl<'a, V, I> IndexSliceMut<'a, I> for SliceDropMut<'a, V>
where
    I: std::slice::SliceIndex<[u8], Output = [u8]> + ScaleRange,
{
    type Output = SliceDropMut<'a, V>;
    fn get_mut(&'a mut self, i: I) -> Self::Output {
        SliceDropMut {
            data: self.data.get_mut(i),
        }
    }
}

/// Convert a `&mut [T]` to a `SliceDropMut`.
impl<'a, T, V> From<&'a mut [T]> for SliceDropMut<'a, V>
where
    T: Elem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a mut [T]) -> SliceDropMut<'a, V> {
        SliceDropMut::from_slice(s)
    }
}

impl<'a, V> From<SliceDropMut<'a, V>> for SliceDrop<'a, V> {
    #[inline]
    fn from(s: SliceDropMut<'a, V>) -> SliceDrop<'a, V> {
        SliceDrop {
            data: SliceCopy::from(s.data),
        }
    }
}

impl<'b, 'a: 'b, V> From<&'b SliceDropMut<'a, V>> for SliceDrop<'b, V> {
    #[inline]
    fn from(s: &'b SliceDropMut<'a, V>) -> SliceDrop<'b, V> {
        unsafe {
            SliceDrop::from_raw_parts(
                s.data.data,
                s.data.element_size,
                s.data.element_type_id,
                s.data.vtable.as_ref(),
            )
        }
    }
}
