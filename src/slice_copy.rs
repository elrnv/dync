use std::{
    any::{Any, TypeId},
    mem::size_of,
    slice,
};

use crate::value::*;
use crate::Elem;

/*
 * Immutable slice
 */

#[derive(Clone)]
pub struct SliceCopy<'a, V = ()> {
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
    pub unsafe fn from_raw_parts(
        data: &'a [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: &'a V,
    ) -> Self {
        SliceCopy {
            data,
            element_size,
            element_type_id,
            vtable: VTableRef::Ref(vtable),
        }
    }

    /// Construct a `SliceCopy` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &[T]) -> Self
    where
        V: VTable<T>,
    {
        // This is safe since `Elem` is `Copy`.
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
            vtable: VTableRef::Owned(Box::new(V::build_vtable())),
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
    pub fn append_copy_to_vec<'b, T: Elem>(&self, vec: &'b mut Vec<T>) -> Option<&'b mut Vec<T>> {
        let iter = self.iter_as()?;
        vec.extend(iter);
        Some(vec)
    }

    /// Copies contents of `self` into the given `Vec`.
    #[inline]
    pub fn copy_into_vec<T: Elem>(&self) -> Option<Vec<T>> {
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
    pub fn get<T: Elem>(&self, i: usize) -> Option<&T> {
        assert!(i < self.len());
        let ptr = self.check::<T>()?.data.as_ptr() as *const T;
        Some(unsafe { &*ptr.add(i) })
    }

    /*
     * Value API. This allows users to manipulate contained data without knowing the element type.
     */

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn value(&self, i: usize) -> CopyValueRef {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe { CopyValueRef::from_raw_parts(self.get_bytes(i), self.element_type_id()) }
    }

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
    pub fn iter(&self) -> impl Iterator<Item = CopyValueRef> {
        self.byte_chunks().map(move |bytes| unsafe {
            CopyValueRef::from_raw_parts(bytes, self.element_type_id())
        })
    }

    /*
     * Advanced methods to probe buffer internals.
     */

    /// Get a `const` reference to the byte slice of the `i`'th element of the buffer.
    #[inline]
    pub(crate) fn get_bytes(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.len());
        let element_size = self.element_size();
        &self.data[i * element_size..(i + 1) * element_size]
    }

    /// Iterate over chunks type sized chunks of bytes without interpreting them.
    #[inline]
    pub(crate) fn byte_chunks(&self) -> impl Iterator<Item = &[u8]> {
        let chunk_size = self.element_size();
        self.data.chunks_exact(chunk_size)
    }
}

/// Convert a `&[T]` to a `SliceCopy`.
impl<'a, T, V> From<&'a [T]> for SliceCopy<'a, V>
where
    T: Elem,
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

pub struct SliceCopyMut<'a, V = ()> {
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
    pub unsafe fn from_raw_parts(
        data: &'a mut [u8],
        element_size: usize,
        element_type_id: TypeId,
        vtable: &'a V,
    ) -> SliceCopyMut<'a, V> {
        SliceCopyMut {
            data,
            element_size,
            element_type_id,
            vtable: VTableRef::Ref(vtable),
        }
    }

    /// Construct a `SliceCopyMut` from a given typed slice by reusing the provided memory.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &'a mut [T]) -> SliceCopyMut<'a, V>
    where
        V: VTable<T>,
    {
        // This is safe since `Elem` is `Copy`.
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
            vtable: VTableRef::Owned(Box::new(V::build_vtable())),
        }
    }

    /// Copy data from a given slice into the current slice.
    ///
    /// # Panics
    ///
    /// This function will panic if the given slice has as different size than current.
    #[inline]
    pub fn copy_from_slice<T: Elem>(&mut self, slice: &[T]) -> Option<&mut Self> {
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
    pub fn append_copy_to_vec<'b, T: Elem>(&self, vec: &'b mut Vec<T>) -> Option<&'b mut Vec<T>> {
        let slice = SliceCopy::from(self);
        vec.extend(slice.iter_as()?);
        Some(vec)
    }

    /// Copies contents of `self` into the given `Vec`.
    #[inline]
    pub fn copy_into_vec<T: Elem>(self) -> Option<Vec<T>> {
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
    pub fn get<T: Elem>(&mut self, i: usize) -> Option<&mut T> {
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

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn value(&mut self, i: usize) -> CopyValueMut {
        debug_assert!(i < self.len());
        let element_size = self.element_size();
        let element_type_id = self.element_type_id();

        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe {
            CopyValueMut::from_raw_parts(
                &mut self.data[i * element_size..(i + 1) * element_size],
                element_type_id,
            )
        }
    }

    /// Return an iterator over untyped value references stored in this slice.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    #[inline]
    pub fn iter(&mut self) -> impl Iterator<Item = CopyValueMut> {
        let Self {
            ref mut data,
            element_size,
            element_type_id,
            ..
        } = *self;
        data.chunks_exact_mut(element_size)
            .map(move |bytes| unsafe { CopyValueMut::from_raw_parts(bytes, element_type_id) })
    }
}

/// Convert a `&mut [T]` to a `SliceCopyMut`.
impl<'a, T, V> From<&'a mut [T]> for SliceCopyMut<'a, V>
where
    T: Elem,
    V: VTable<T>,
{
    #[inline]
    fn from(s: &'a mut [T]) -> SliceCopyMut<'a, V> {
        SliceCopyMut::from_slice(s)
    }
}

impl<'a, V> From<SliceCopyMut<'a, V>> for SliceCopy<'a, V> {
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

impl<'b, 'a: 'b, V> From<&'b SliceCopyMut<'a, V>> for SliceCopy<'b, V> {
    #[inline]
    fn from(s: &'b SliceCopyMut<'a, V>) -> SliceCopy<'b, V> {
        unsafe {
            SliceCopy::from_raw_parts(s.data, s.element_size, s.element_type_id, s.vtable.as_ref())
        }
    }
}
