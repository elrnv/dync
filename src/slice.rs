use std::{
    any::{Any, TypeId},
    fmt,
    mem::MaybeUninit,
    slice,
};

use crate::elem::ElemInfo;
use crate::index_slice::*;
use crate::slice_copy::*;
use crate::traits::*;
use crate::value::*;
use crate::vtable::*;
use crate::Elem;
use crate::ElementBytes;

/*
 * Immutable slice
 */

#[derive(Clone)]
pub struct Slice<'a, V>
where
    V: ?Sized,
{
    pub(crate) data: SliceCopy<'a, V>,
}

pub type SliceDrop<'a> = Slice<'a, DropVTable>;

impl<'a, V: ?Sized + HasDrop + HasPartialEq> PartialEq for Slice<'a, V> {
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(this, that)| this == that)
    }
}

impl<'a, V: ?Sized + HasDrop + HasDebug> fmt::Debug for Slice<'a, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, V: HasDrop> Slice<'a, V> {
    /// Construct a `Slice` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        Slice {
            data: unsafe { SliceCopy::from_slice_non_copy(slice) },
        }
    }
}

impl<'a, V: ?Sized + HasDrop> Slice<'a, V> {
    /// Convert this collection into its raw components.
    ///
    /// This function exists mainly to enable the `into_dyn` macro until `CoerceUnsized` is
    /// stabilized.
    #[inline]
    pub fn into_raw_parts(self) -> (&'a [MaybeUninit<u8>], ElemInfo, VTableRef<'a, V>) {
        let SliceCopy { data, elem, vtable } = self.data;
        (data, elem, vtable)
    }

    /// Construct a `Slice` from raw bytes and type metadata.
    ///
    /// # Safety
    ///
    /// Almost exclusively the only inputs that are safe here are the ones returned by
    /// `into_raw_parts`.
    ///
    /// This function should not be used other than in internal APIs. It exists to enable the
    /// `into_dyn` macro until `CoerceUsize` is stabilized.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: &'a [MaybeUninit<u8>],
        elem: ElemInfo,
        vtable: impl Into<VTableRef<'a, V>>,
    ) -> Self {
        Slice {
            data: SliceCopy::from_raw_parts(data, elem, vtable),
        }
    }

    /// Upcast the `Slice` into a more general base `Slice`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> Slice<'a, U>
    where
        V: Clone,
    {
        Slice {
            data: self.data.upcast(), //_with(|v: V| (v.0, U::from(v.1))),
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
    pub fn reborrow(&self) -> Slice<V> {
        Slice {
            data: self.data.reborrow(),
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
    /// Returns `None` if the given type `T` doesn't match the internal.
    ///
    /// # Examples
    /// ```
    /// use dync::SliceDrop;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf: SliceDrop = vec.as_slice().into();
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
    /// let buf: SliceDrop = vec.as_slice().into();
    /// for (i, val) in buf.iter().enumerate() {
    ///     assert_eq!(val.downcast::<f32>().unwrap(), &vec[i]);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ValueRef<V>> {
        self.data.byte_chunks().map(move |bytes| unsafe {
            ValueRef::from_raw_parts(
                bytes,
                self.element_type_id(),
                self.data.elem.alignment,
                self.data.vtable.as_ref(),
            )
        })
    }

    // TODO: Determine if we can instead implement IntoIterator or explain why not and silence the clippy warning.
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = ValueRef<'a, V>>
    where
        V: Clone,
    {
        let SliceCopy { data, elem, vtable } = self.data;
        data.chunks_exact(elem.num_bytes())
            .map(move |bytes| unsafe {
                ValueRef::from_raw_parts(bytes, elem.type_id, elem.alignment, vtable.clone())
            })
    }

    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = Slice<V>> {
        self.data
            .chunks_exact(chunk_size)
            .map(|data| Slice { data })
    }

    #[inline]
    pub fn split_at(&self, mid: usize) -> (Slice<V>, Slice<V>) {
        let (l, r) = self.data.split_at(mid);
        (Slice { data: l }, Slice { data: r })
    }

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> ValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueRef::from_raw_parts(
                self.index_byte_slice(i),
                self.element_type_id(),
                self.data.elem.alignment,
                self.data.vtable.as_ref(),
            )
        }
    }

    /// Get an immutable subslice representing the given range of indices.
    #[inline]
    pub fn subslice<I>(&self, i: I) -> Slice<V>
    where
        I: std::slice::SliceIndex<[MaybeUninit<u8>], Output = [MaybeUninit<u8>]> + ScaleRange,
    {
        Slice {
            data: self.data.subslice(i),
        }
    }

    /// Convert this slice into an immutable subslice representing the given range of indices.
    #[inline]
    pub fn into_subslice<I>(self, i: I) -> Slice<'a, V>
    where
        I: std::slice::SliceIndex<[MaybeUninit<u8>], Output = [MaybeUninit<u8>]> + ScaleRange,
    {
        Slice {
            data: self.data.into_subslice(i),
        }
    }

    /*
     * Advanced methods to probe buffer internals.
     */

    /// Get a `const` reference to the byte slice of the `i`'th element of the buffer.
    #[inline]
    pub(crate) fn index_byte_slice(&self, i: usize) -> &[MaybeUninit<u8>] {
        self.data.index_byte_slice(i)
    }
}

impl<'a, V: ?Sized + HasClone + HasDrop> Slice<'a, V> {
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
        // NOTE: vec cannot be captured by closure if it's also mutably borrowed.
        #[allow(clippy::manual_map)]
        match self.append_clone_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }
}

/// Convert a `&[T]` to a `Slice`.
impl<'a, T, V> From<&'a [T]> for Slice<'a, V>
where
    T: Elem,
    V: VTable<T> + HasDrop,
{
    #[inline]
    fn from(s: &'a [T]) -> Slice<'a, V> {
        Slice::from_slice(s)
    }
}

unsafe impl<'a, V: ?Sized + HasDrop + HasSend> Send for Slice<'a, V> {}
unsafe impl<'a, V: ?Sized + HasDrop + HasSync> Sync for Slice<'a, V> {}

/*
 * Mutable Slice
 */

pub struct SliceMut<'a, V>
where
    V: ?Sized,
{
    pub(crate) data: SliceCopyMut<'a, V>,
}

pub type SliceMutDrop<'a> = SliceMut<'a, DropVTable>;

impl<'a, V: ?Sized + HasDrop + HasPartialEq> PartialEq for SliceMut<'a, V> {
    fn eq(&self, other: &Self) -> bool {
        self.reborrow()
            .iter()
            .zip(other.reborrow().iter())
            .all(|(this, that)| this == that)
    }
}

impl<'a, V: ?Sized + HasDrop + HasDebug> fmt::Debug for SliceMut<'a, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.reborrow().iter()).finish()
    }
}

impl<'a, V: HasDrop> SliceMut<'a, V> {
    /// Construct a `SliceMut` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &'a mut [T]) -> SliceMut<'a, V>
    where
        V: VTable<T>,
    {
        SliceMut {
            data: unsafe { SliceCopyMut::from_slice_non_copy(slice) },
        }
    }
}

impl<'a, V: ?Sized + HasDrop> SliceMut<'a, V> {
    /// Convert this collection into its raw components.
    ///
    /// This function exists mainly to enable the `into_dyn` macro until `CoerceUnsized` is
    /// stabilized.
    #[inline]
    pub fn into_raw_parts(self) -> (&'a [MaybeUninit<u8>], ElemInfo, VTableRef<'a, V>) {
        let SliceCopyMut { data, elem, vtable } = self.data;
        (data, elem, vtable)
    }

    /// Construct a `SliceMut` from raw bytes and type metadata.
    ///
    /// # Safety
    ///
    /// Almost exclusively the only inputs that are safe here are the ones returned by
    /// `into_raw_parts`.
    ///
    /// This function should not be used other than in internal APIs. It exists to enable the
    /// `into_dyn` macro until `CoerceUsize` is stabilized.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: &'a mut [MaybeUninit<u8>],
        elem: ElemInfo,
        vtable: impl Into<VTableRef<'a, V>>,
    ) -> SliceMut<'a, V> {
        SliceMut {
            data: SliceCopyMut::from_raw_parts(data, elem, vtable),
        }
    }

    /// Upcast the `SliceMut` into a more general base `SliceMut`.
    ///
    /// This function converts the underlying virtual function table into a subset of the existing
    #[inline]
    pub fn upcast<U: From<V>>(self) -> SliceMut<'a, U>
    where
        V: Clone,
    {
        SliceMut {
            data: self.data.upcast(), //_with(|v: (DropFn, V)| (v.0, U::from(v.1))),
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

    /// Construct a clone of the current slice with a reduced lifetime.
    ///
    /// This is equivalent to calling `subslice` with the entire range.
    #[inline]
    pub fn reborrow(&self) -> Slice<V> {
        Slice {
            data: self.data.reborrow(),
        }
    }

    /// Construct a clone of the current slice with a reduced lifetime.
    ///
    /// This is equivalent to calling `subslice_mut` with the entire range.
    #[inline]
    pub fn reborrow_mut(&mut self) -> SliceMut<V> {
        SliceMut {
            data: self.data.reborrow_mut(),
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
    /// use dync::SliceMutDrop;
    /// let mut vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let mut buf: SliceMutDrop = vec.as_mut_slice().into();
    /// for val in buf.iter_as::<f32>().unwrap() {
    ///     *val += 1.0_f32;
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&mut self) -> Option<slice::IterMut<T>> {
        self.as_slice::<T>().map(|x| x.iter_mut())
    }

    /// Convert this `SliceMut` into a typed slice.
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
    /// let mut buf: SliceMutDrop = vec.as_mut_slice().into();
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
    /// let mut buf: SliceMutDrop = vec.as_mut_slice().into();
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
            elem,
            ref vtable,
        } = self.data;
        data.chunks_exact_mut(elem.num_bytes())
            .map(move |bytes| unsafe {
                ValueMut::from_raw_parts(bytes, elem.type_id, elem.alignment, vtable.as_ref())
            })
    }

    // TODO: Determine if we can instead implement IntoIterator or explain why not and silence the clippy warning.
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = ValueMut<'a, V>>
    where
        V: Clone,
    {
        let SliceCopyMut { data, elem, vtable } = self.data;
        data.chunks_exact_mut(elem.num_bytes())
            .map(move |bytes| unsafe {
                ValueMut::from_raw_parts(bytes, elem.type_id, elem.alignment, vtable.clone())
            })
    }

    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = Slice<V>> {
        self.data
            .chunks_exact(chunk_size)
            .map(|data| Slice { data })
    }

    #[inline]
    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> impl Iterator<Item = SliceMut<V>> {
        self.data
            .chunks_exact_mut(chunk_size)
            .map(|data| SliceMut { data })
    }

    #[inline]
    pub fn split_at(&mut self, mid: usize) -> (SliceMut<V>, SliceMut<V>) {
        let (l, r) = self.data.split_at(mid);
        (SliceMut { data: l }, SliceMut { data: r })
    }

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> ValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueRef::from_raw_parts(
                self.data.index_byte_slice(i),
                self.element_type_id(),
                self.data.elem.alignment,
                self.data.vtable.as_ref(),
            )
        }
    }

    /// Get a mutable reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> ValueMut<V> {
        let CopyValueMut {
            bytes,
            type_id,
            alignment,
            vtable,
        } = self.data.get_mut(i);

        ValueMut {
            bytes,
            type_id,
            alignment,
            vtable,
        }
    }

    /// Get an immutable subslice from the given range of indices.
    #[inline]
    pub fn subslice<I>(&self, i: I) -> Slice<V>
    where
        I: std::slice::SliceIndex<[MaybeUninit<u8>], Output = [MaybeUninit<u8>]> + ScaleRange,
    {
        Slice {
            data: self.data.subslice(i),
        }
    }

    /// Get a mutable subslice from the given range of indices.
    #[inline]
    pub fn subslice_mut<I>(&mut self, i: I) -> SliceMut<V>
    where
        I: std::slice::SliceIndex<[MaybeUninit<u8>], Output = [MaybeUninit<u8>]> + ScaleRange,
    {
        SliceMut {
            data: self.data.subslice_mut(i),
        }
    }

    /// Convert this slice into a mutable subslice from the given range of indices.
    #[inline]
    pub fn into_subslice<I>(self, i: I) -> SliceMut<'a, V>
    where
        I: std::slice::SliceIndex<[MaybeUninit<u8>], Output = [MaybeUninit<u8>]> + ScaleRange,
    {
        SliceMut {
            data: self.data.into_subslice(i),
        }
    }
}

impl<'a, V: HasDrop + HasClone> SliceMut<'a, V> {
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
        Slice::from(self).append_clone_to_vec(vec)?;
        Some(vec)
    }

    /// Clones contents of `self` into the given `Vec`.
    #[inline]
    pub fn clone_into_vec<T: Elem + Clone>(self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        // NOTE: vec cannot be captured by closure if it's also mutably borrowed.
        #[allow(clippy::manual_map)]
        match self.append_clone_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }
}

/// Convert a `&mut [T]` to a `SliceMut`.
impl<'a, T, V> From<&'a mut [T]> for SliceMut<'a, V>
where
    T: Elem,
    V: VTable<T> + HasDrop,
{
    #[inline]
    fn from(s: &'a mut [T]) -> SliceMut<'a, V> {
        SliceMut::from_slice(s)
    }
}

impl<'a, V: ?Sized> From<SliceMut<'a, V>> for Slice<'a, V> {
    #[inline]
    fn from(s: SliceMut<'a, V>) -> Slice<'a, V> {
        Slice {
            data: SliceCopy::from(s.data),
        }
    }
}

impl<'b, 'a: 'b, V: ?Sized + HasDrop> From<&'b SliceMut<'a, V>> for Slice<'b, V> {
    #[inline]
    fn from(s: &'b SliceMut<'a, V>) -> Slice<'b, V> {
        unsafe { Slice::from_raw_parts(s.data.data, s.data.elem, s.data.vtable.as_ref()) }
    }
}

unsafe impl<'a, V: ?Sized + HasDrop + HasSend> Send for SliceMut<'a, V> {}
unsafe impl<'a, V: ?Sized + HasDrop + HasSync> Sync for SliceMut<'a, V> {}
