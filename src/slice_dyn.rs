use std::{
    any::{Any, TypeId},
    slice,
};

use crate::slice_copy::*;
use crate::traits::*;
use crate::value::*;
use crate::Elem;

/*
 * Immutable slice
 */

#[derive(Clone)]
pub struct SliceDyn<'a, V> {
    pub(crate) data: SliceCopy<'a, (DropFn, V)>,
}

impl<'a, V> SliceDyn<'a, V> {
    pub(crate) unsafe fn from_raw_parts(
        data: &'a [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: impl Into<VTableRef<'a, (DropFn, V)>>,
    ) -> Self {
        SliceDyn {
            data: SliceCopy::from_raw_parts(data, element_size, element_type_id, vtable),
        }
    }

    /// Construct a `SliceDyn` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        SliceDyn {
            data: unsafe { SliceCopy::from_slice_non_copy(slice) },
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
    /// use dync::SliceDyn;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf = SliceDyn::<()>::from_slice(vec.as_slice());
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

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> ValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            ValueRef::from_raw_parts(
                self.get_bytes(i),
                self.element_type_id(),
                self.data.vtable.as_ref(),
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
    /// use dync::SliceDyn;
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf = SliceDyn::<()>::from(vec.as_slice());
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

    pub fn split_at(&self, mid: usize) -> (SliceDyn<V>, SliceDyn<V>) {
        let (l, r) = self.data.split_at(mid);
        (SliceDyn { data: l }, SliceDyn { data: r })
    }

    /*
     * Advanced methods to probe buffer internals.
     */

    /// Get a `const` reference to the byte slice of the `i`'th element of the buffer.
    #[inline]
    pub(crate) fn get_bytes(&self, i: usize) -> &[u8] {
        self.data.get_bytes(i)
    }
}

impl<'a, V: HasClone> SliceDyn<'a, V> {
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

/// Convert a `&[T]` to a `SliceDyn`.
impl<'a, T, V> From<&'a [T]> for SliceDyn<'a, V>
where
    T: Elem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a [T]) -> SliceDyn<'a, V> {
        SliceDyn::from_slice(s)
    }
}

/*
 * Mutable Slice
 */

pub struct SliceDynMut<'a, V> {
    pub(crate) data: SliceCopyMut<'a, (DropFn, V)>,
}

impl<'a, V> SliceDynMut<'a, V> {
    pub(crate) unsafe fn from_raw_parts(
        data: &'a mut [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: impl Into<VTableRef<'a, (DropFn, V)>>,
    ) -> SliceDynMut<'a, V> {
        SliceDynMut {
            data: SliceCopyMut::from_raw_parts(data, element_size, element_type_id, vtable),
        }
    }

    /// Construct a `SliceDynMut` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &'a mut [T]) -> SliceDynMut<'a, V>
    where
        V: VTable<T>,
    {
        SliceDynMut {
            data: unsafe { SliceCopyMut::from_slice_non_copy(slice) },
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
    /// use dync::SliceDynMut;
    /// let mut vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let mut buf = SliceDynMut::<()>::from(vec.as_mut_slice());
    /// for val in buf.iter_as::<f32>().unwrap() {
    ///     *val += 1.0_f32;
    /// }
    /// ```
    #[inline]
    pub fn iter_as<T: Any>(&mut self) -> Option<slice::IterMut<T>> {
        self.as_slice::<T>().map(|x| x.iter_mut())
    }

    /// Convert this `SliceDynMut` into a typed slice.
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
    /// let mut buf = SliceDynMut::<()>::from_slice(vec.as_mut_slice());
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
    /// let mut buf = SliceDynMut::<()>::from_slice(vec.as_mut_slice());
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

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn get(&mut self, i: usize) -> ValueMut<V> {
        let CopyValueMut {
            bytes,
            type_id,
            vtable,
        } = self.data.get(i);

        ValueMut {
            bytes,
            type_id,
            vtable,
        }
    }

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

    pub fn split_at(&mut self, mid: usize) -> (SliceDynMut<V>, SliceDynMut<V>) {
        let (l, r) = self.data.split_at(mid);
        (SliceDynMut { data: l }, SliceDynMut { data: r })
    }
}

impl<'a, V: HasClone> SliceDynMut<'a, V> {
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
        SliceDyn::from(self).append_clone_to_vec(vec)?;
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

/// Convert a `&mut [T]` to a `SliceDynMut`.
impl<'a, T, V> From<&'a mut [T]> for SliceDynMut<'a, V>
where
    T: Elem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a mut [T]) -> SliceDynMut<'a, V> {
        SliceDynMut::from_slice(s)
    }
}

impl<'a, V> From<SliceDynMut<'a, V>> for SliceDyn<'a, V> {
    #[inline]
    fn from(s: SliceDynMut<'a, V>) -> SliceDyn<'a, V> {
        SliceDyn {
            data: SliceCopy::from(s.data),
        }
    }
}

impl<'b, 'a: 'b, V> From<&'b SliceDynMut<'a, V>> for SliceDyn<'b, V> {
    #[inline]
    fn from(s: &'b SliceDynMut<'a, V>) -> SliceDyn<'b, V> {
        unsafe {
            SliceDyn::from_raw_parts(
                s.data.data,
                s.data.element_size,
                s.data.element_type_id,
                s.data.vtable.as_ref(),
            )
        }
    }
}
