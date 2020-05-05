#![allow(dead_code)]
use std::{
    any::{Any, TypeId},
    mem::ManuallyDrop,
    slice,
};

use crate::clone_value::*;
use crate::traits::*;
use crate::VecCopy;

pub trait Elem: Any + CloneBytes + DropBytes {}
impl<T> Elem for T where T: Any + CloneBytes + DropBytes {}

/// This container is a WIP, not to be used in production.
pub struct VecClone {
    data: ManuallyDrop<VecCopy>,
    clone_fn: CloneFn,
    clone_from_fn: CloneFromFn,
    drop_fn: DropFn,
}

impl Clone for VecClone {
    fn clone(&self) -> Self {
        let data_clone = |bytes: &[u8]| {
            let mut new_data = bytes.to_vec();
            self.data
                .byte_chunks()
                .zip(new_data.chunks_exact_mut(self.data.element_size()))
                .for_each(|(src, dst)| unsafe { (self.clone_from_fn)(dst, src) });
            new_data
        };
        VecClone {
            data: ManuallyDrop::new(self.data.clone_with(data_clone)),
            clone_fn: self.clone_fn.clone(),
            clone_from_fn: self.clone_from_fn.clone(),
            drop_fn: self.drop_fn.clone(),
        }
    }
}

impl Drop for VecClone {
    fn drop(&mut self) {
        unsafe {
            for elem_bytes in self.data.byte_chunks_mut() {
                (self.drop_fn)(elem_bytes);
            }
        }
    }
}

impl VecClone {
    /// Construct an empty vector with a specific pointed-to element type.
    #[inline]
    pub fn with_type<T: Elem>() -> Self {
        VecClone {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::with_type_non_copy::<T>() }),
            clone_fn: T::clone_bytes,
            clone_from_fn: T::clone_from_bytes,
            drop_fn: T::drop_bytes,
        }
    }

    /// Construct a vector with the same type as the given vector without copying its data.
    #[inline]
    pub fn with_type_from(other: &VecClone) -> Self {
        VecClone {
            data: ManuallyDrop::new(VecCopy::with_type_from(&other.data)),
            clone_fn: other.clone_fn.clone(),
            clone_from_fn: other.clone_from_fn.clone(),
            drop_fn: other.drop_fn.clone(),
        }
    }

    /// Construct an empty vector with a capacity for a given number of typed pointed-to elements.
    #[inline]
    pub fn with_capacity<T: Elem>(n: usize) -> Self {
        VecClone {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::with_capacity_non_copy::<T>(n) }),
            clone_fn: T::clone_bytes,
            clone_from_fn: T::clone_from_bytes,
            drop_fn: T::drop_bytes,
        }
    }

    /// Construct a typed `DataBuffer` with a given size and filled with the specified default
    /// value.
    #[inline]
    pub fn with_size<T: Elem>(n: usize, def: T) -> Self {
        VecClone {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_vec_non_copy(vec![def; n]) }),
            clone_fn: T::clone_bytes,
            clone_from_fn: T::clone_from_bytes,
            drop_fn: T::drop_bytes,
        }
    }

    /// Construct a `VecClone` from a given `Vec` reusing the space already allocated by the given
    /// vector.
    pub fn from_vec<T: Elem>(vec: Vec<T>) -> Self {
        VecClone {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_vec_non_copy(vec) }),
            clone_fn: T::clone_bytes,
            clone_from_fn: T::clone_from_bytes,
            drop_fn: T::drop_bytes,
        }
    }

    /// Construct a buffer from a given slice by cloning the data.
    #[inline]
    pub fn from_slice<T: Elem>(slice: &[T]) -> Self {
        VecClone {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_slice_non_copy::<T>(slice) }),
            clone_fn: T::clone_bytes,
            clone_from_fn: T::clone_from_bytes,
            drop_fn: T::drop_bytes,
        }
    }

    /// Resizes the buffer in-place to store `new_len` elements and returns an optional
    /// mutable reference to `Self`.
    ///
    /// If `value` does not correspond to the underlying element type, then `None` is returned and the
    /// buffer is left unchanged.
    ///
    /// This function has the similar properties to `Vec::resize`.
    #[inline]
    pub fn resize<T: Elem>(&mut self, new_len: usize, value: T) -> Option<&mut Self> {
        self.check_ref::<T>()?;
        let size_t = std::mem::size_of::<T>();
        if new_len >= self.len() {
            let diff = new_len - self.len();
            self.data.reserve_bytes(diff * size_t);
            for _ in 0..diff {
                self.data.push(value.clone());
            }
        } else {
            // Drop trailing elements manually.
            unsafe {
                for bytes in self.data.byte_chunks_mut().skip(new_len) {
                    (self.drop_fn)(bytes);
                }
            }
            // Truncate data
            self.data.data.resize(new_len * size_t, 0);
        }
        Some(self)
    }

    /// Clear the data buffer without destroying its type information.
    #[inline]
    pub fn clear(&mut self) {
        // Drop all elements manually.
        unsafe {
            for bytes in self.data.byte_chunks_mut() {
                (self.drop_fn)(bytes);
            }
        }
        self.data.data.clear();
    }

    /// Fill the current buffer with clones of the given value.
    ///
    /// The size of the buffer is left unchanged. If the given type doesn't match the
    /// internal type, `None` is returned, otherwise a mutable reference to the modified buffer is
    /// returned.
    #[inline]
    pub fn fill<T: Elem>(&mut self, def: T) -> Option<&mut Self> {
        for v in self.iter_mut::<T>()? {
            *v = def.clone();
        }
        Some(self)
    }

    /// Add an element to this buffer.
    ///
    /// If the type of the given element coincides with the type stored by this buffer,
    /// then the modified buffer is returned via a mutable reference.  Otherwise, `None` is
    /// returned.
    #[inline]
    pub fn push<T: Elem>(&mut self, element: T) -> Option<&mut Self> {
        if let Some(_) = self.data.push(element) {
            Some(self)
        } else {
            None
        }
    }

    /// Check if the current buffer contains elements of the specified type. Returns `Some(self)`
    /// if the type matches and `None` otherwise.
    #[inline]
    pub fn check<T: Elem>(self) -> Option<Self> {
        if let Some(_) = self.data.check_ref::<T>() {
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
    pub fn check_mut<'a, T: Elem>(&'a mut self) -> Option<&'a mut Self> {
        if let Some(_) = self.data.check_mut::<T>() {
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
    pub fn iter<'a, T: Elem>(&'a self) -> Option<slice::Iter<T>> {
        self.data.iter::<T>()
    }

    /// Return an iterator to a mutable slice representing typed data.
    ///
    /// Returns `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn iter_mut<'a, T: Elem>(&'a mut self) -> Option<slice::IterMut<T>> {
        self.data.iter_mut::<T>()
    }

    /// Append cloned items from this buffer to a given `Vec`.
    ///
    /// Return the mutable reference `Some(vec)` if type matched the internal type and
    /// `None` otherwise.
    #[inline]
    pub fn append_to_vec<'a, T: Elem>(&self, vec: &'a mut Vec<T>) -> Option<&'a mut Vec<T>> {
        let slice = self.as_slice()?;
        // Only allocate once we have confirmed that the given `T` matches to avoid unnecessary
        // overhead.
        vec.reserve(self.len());
        vec.extend_from_slice(slice);
        Some(vec)
    }

    /// Clones contents of `self` into the given `Vec`.
    #[inline]
    pub fn clone_into_vec<T: Elem>(&self) -> Option<Vec<T>> {
        let mut vec = Vec::new();
        match self.append_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
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
    pub fn as_slice<T: Elem>(&self) -> Option<&[T]> {
        self.data.as_slice()
    }

    /// Convert this buffer into a typed mutable slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_mut_slice<T: Elem>(&mut self) -> Option<&mut [T]> {
        self.data.as_mut_slice()
    }

    /// Get a `const` reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_ref<T: Elem>(&self, i: usize) -> Option<&T> {
        self.data.get_ref::<T>(i)
    }

    /// Get a mutable reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_mut<T: Elem>(&mut self, i: usize) -> Option<&mut T> {
        self.data.get_mut::<T>(i)
    }

    /// Move bytes to this buffer.
    ///
    /// The given buffer must have the same underlying type as `self`.
    #[inline]
    pub fn append(&mut self, buf: &mut VecClone) -> Option<&mut Self> {
        // It is sufficient to move the bytes, no clones or drops are necessary here.
        if let Some(_) = self.data.append(&mut buf.data) {
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

    /// Get a reference to a value stored in this container at index `i`.
    #[inline]
    pub fn value_ref(&self, i: usize) -> CloneValueRef {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe { CloneValueRef::from_raw_parts(self.data.get_bytes(i), self.element_type_id()) }
    }

    /// Get a mutable reference to a value stored in this container at index `i`.
    #[inline]
    pub fn value_mut(&mut self, i: usize) -> CloneValueMut {
        debug_assert!(i < self.len());
        let Self {
            data,
            clone_from_fn,
            ..
        } = self;
        let type_id = data.element_type_id();
        // Safety is guaranteed here by the value API.
        unsafe { CloneValueMut::from_raw_parts(data.get_bytes_mut(i), type_id, *clone_from_fn) }
    }

    /// Return an iterator over untyped value references stored in this buffer.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is
    /// needed for each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use dync::vec_clone::*;
    /// use std::rc::Rc;
    /// let vec: Vec<_> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43].into_iter().map(Rc::new).collect();
    /// let mut buf = VecClone::from(vec); // Convert into VecCLone
    ///
    /// // Compute the sum through an Rc pointer which is Clone by not Copy.
    /// let mut sum = Rc::new(0.0_f32);
    /// for val in buf.iter_value_ref() {
    ///     *Rc::get_mut(&mut sum).unwrap() += **val.downcast::<Rc<f32>>().unwrap();
    /// }
    /// assert!((*sum - 77.44).abs() < 0.000001);
    /// ```
    #[inline]
    pub fn iter_value_ref<'a>(&'a self) -> impl Iterator<Item = CloneValueRef<'a>> + 'a {
        let &Self { ref data, .. } = self;
        let VecCopy {
            data,
            element_size,
            element_type_id,
        } = &**data;
        data.chunks_exact(*element_size)
            .map(move |bytes| unsafe { CloneValueRef::from_raw_parts(bytes, *element_type_id) })
    }

    /// Return an iterator over mutable untyped value references stored in this buffer.
    ///
    /// In contrast to `iter_mut`, this function defers downcasting on a per element basis.  As a
    /// result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    ///
    /// # Examples
    /// ```
    /// use dync::clone_value::*;
    /// use dync::vec_clone::*;
    /// use std::rc::Rc;
    /// let vec: Vec<_> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43].into_iter().map(Rc::new).collect();
    /// let mut buf = VecClone::from(vec.clone()); // Convert into buffer
    ///
    /// // Overwrite all vallues in buf to the following Rc.
    /// let rc = Rc::new(100.0f32);
    /// for mut val in buf.iter_value_mut() {
    ///     val.clone_from(CloneValueRef::new(&rc));
    /// }
    ///
    /// // As a result the data in `rc` has been referenced 6 times.
    /// assert_eq!(Rc::strong_count(&rc), 6);
    /// assert_eq!(buf.into_vec::<Rc<f32>>().unwrap(), vec![rc; 5]);
    /// ```
    #[inline]
    pub fn iter_value_mut<'a>(&'a mut self) -> impl Iterator<Item = CloneValueMut<'a>> + 'a {
        let &mut Self {
            ref mut data,
            clone_from_fn,
            ..
        } = self;
        let VecCopy {
            data,
            element_size,
            element_type_id,
        } = &mut **data;
        data.chunks_exact_mut(*element_size)
            .map(move |bytes| unsafe {
                CloneValueMut::from_raw_parts(bytes, *element_type_id, clone_from_fn)
            })
    }
}

/// Convert a `Vec` to a buffer.
impl<T: Elem> From<Vec<T>> for VecClone {
    #[inline]
    fn from(vec: Vec<T>) -> VecClone {
        VecClone::from_vec(vec)
    }
}

/// Convert a slice to a `VecClone`.
impl<'a, T: Elem> From<&'a [T]> for VecClone {
    #[inline]
    fn from(slice: &'a [T]) -> VecClone {
        VecClone::from_slice(slice)
    }
}

/// Convert a buffer to a `Vec` with an option to fail.
impl<T: Elem> Into<Option<Vec<T>>> for VecClone {
    #[inline]
    fn into(self) -> Option<Vec<T>> {
        self.into_vec()
    }
}

impl From<VecCopy> for VecClone {
    fn from(data: VecCopy) -> VecClone {
        // We can downgrade a `VecCopy` collection into a `VecClone` by providing
        // default empty implementations for clone and drop.
        VecClone {
            data: ManuallyDrop::new(data),
            clone_fn: |v: &[u8]| v.to_vec().into_boxed_slice(),
            clone_from_fn: |_, _| {},
            drop_fn: |_| {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use std::rc::Rc;

    #[test]
    fn clone_from_test() {
        //use std::collections::HashSet;
        //use std::rc::Rc;

        //// Let's create a collection of `Rc`s.
        //let vec_rc: Vec<_> = vec![1.0_f32, 23.0, 0.01, 42.0, 23.0, 1.0]
        //    .into_iter()
        //    .map(Rc::new)
        //    .collect();
        //let buf = VecClone::from(vec_rc.clone()); // Clone into VecClone

        //// Construct a hashset of unique values from the VecClone.
        //let mut hashset: HashSet<RcValue> = HashSet::new();

        //for rc in vec_rc.iter().take(4) {
        //    assert!(hashset.insert(Rc::clone(rc).into()));
        //}

        //assert!(!hashset.insert(Rc::clone(&vec_rc[4]).into()));
        //assert!(!hashset.insert(Rc::clone(&vec_rc[5]).into()));

        //assert_eq!(hashset.len(), 4);
        //assert!(hashset.contains(&Rc::new(1.0f32).into()));
        //assert!(hashset.contains(&Rc::new(23.0f32).into()));
        //assert!(hashset.contains(&Rc::new(0.01f32).into()));
        //assert!(hashset.contains(&Rc::new(42.0f32).into()));
        //assert!(!hashset.contains(&Rc::new(42.0).into()));
    }

    #[test]
    fn iter_value_ref() {
        use std::rc::Rc;
        let vec: Vec<_> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43]
            .into_iter()
            .map(Rc::new)
            .collect();
        {
            let buf = VecClone::from(vec.clone()); // Convert into buffer
            let orig = Rc::new(100.0f32);
            let mut rc = Rc::clone(&orig);
            assert_eq!(Rc::strong_count(&rc), 2);
            for val in buf.iter_value_ref() {
                CloneValueMut::new(&mut rc).clone_from(val);
            }
            assert_eq!(Rc::strong_count(&orig), 1);
            assert_eq!(Rc::strong_count(&rc), 3);
            assert_eq!(Rc::strong_count(&vec[4]), 3);
            assert!(vec.iter().take(4).all(|x| Rc::strong_count(x) == 2));
            assert_eq!(rc, Rc::new(11.43));
        }
        assert!(vec.iter().all(|x| Rc::strong_count(x) == 1));
    }

    /// Test various ways to create a `VecClone`.
    #[test]
    fn initialization_test() {
        // Empty typed buffer.
        let a = VecClone::with_type::<Rc<f32>>();
        assert_eq!(a.len(), 0);
        assert_eq!(a.element_type_id(), TypeId::of::<Rc<f32>>());
        assert_eq!(a.byte_capacity(), 0); // Ensure nothing is allocated.

        // Empty buffer typed by the given type id.
        let b = VecClone::with_type_from(&a);
        assert_eq!(b.len(), 0);
        assert_eq!(b.element_type_id(), TypeId::of::<Rc<f32>>());
        assert_eq!(a.byte_capacity(), 0); // Ensure nothing is allocated.

        // Empty typed buffer with a given capacity.
        let a = VecClone::with_capacity::<Rc<f32>>(4);
        assert_eq!(a.len(), 0);
        assert_eq!(a.byte_capacity(), 4 * size_of::<Rc<f32>>());
        assert_eq!(a.element_type_id(), TypeId::of::<Rc<f32>>());
    }

    /// Test resizing a buffer.
    #[test]
    fn resize() {
        let mut a = VecClone::with_type::<Rc<f32>>();

        // Increase the size of a.
        a.resize(3, Rc::new(1.0f32));

        assert_eq!(a.len(), 3);
        for i in 0..3 {
            assert_eq!(a.get_ref::<Rc<f32>>(i).unwrap(), &Rc::new(1.0f32));
        }

        // Truncate a.
        a.resize(2, Rc::new(1.0f32));

        assert_eq!(a.len(), 2);
        for i in 0..2 {
            assert_eq!(a.get_ref::<Rc<f32>>(i).unwrap(), &Rc::new(1.0f32));
        }
    }

    #[test]
    fn data_integrity_u8_test() {
        let vec: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2].into_iter().map(Rc::new).collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u8>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2, 52, 1, 3, 41, 23, 2]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u8>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i16_test() {
        let vec: Vec<Rc<i16>> = vec![1i16, -3, 1002, -231, 32]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i16>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<i16>> = vec![1i16, -3, 1002, -231, 32, 42, -123, 4]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i16>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i32_test() {
        let vec: Vec<Rc<i32>> = vec![1i32, -3, 1002, -231, 32]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<i32>> = vec![1i32, -3, 1002, -231, 32, 42, -123]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_f32_test() {
        let vec: Vec<Rc<f32>> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<f32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<f32>> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43, 2e-1]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<f32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_f64_test() {
        let vec: Vec<Rc<f64>> = vec![1f64, -3.0, 10.02, -23.1, 32e-1]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<f64>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<f64>> = vec![1f64, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<f64>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Foo {
        a: u8,
        b: i64,
        c: f32,
    }

    #[test]
    fn from_empty_vec_test() {
        let vec: Vec<Rc<u32>> = Vec::new();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<String>> = Vec::new();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<String>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<Foo>> = Vec::new();
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<Foo>> = buf.into_vec().unwrap(); // Convert back into vec
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
        let vec: Vec<Rc<Foo>> = vec![Rc::new(f1.clone()), Rc::new(f2.clone())];
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        assert_eq!(Rc::new(f1), buf.get_ref::<Rc<Foo>>(0).unwrap().clone());
        assert_eq!(Rc::new(f2), buf.get_ref::<Rc<Foo>>(1).unwrap().clone());
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
        let buf = VecClone::from(vec.clone()); // Convert into buffer
        assert_eq!(
            &Rc::new("hi".to_string()),
            buf.get_ref::<Rc<String>>(0).unwrap()
        );
        assert_eq!(
            &Rc::new("hello".to_string()),
            buf.get_ref::<Rc<String>>(1).unwrap()
        );
        assert_eq!(
            &Rc::new("goodbye".to_string()),
            buf.get_ref::<Rc<String>>(2).unwrap()
        );
        let nu_vec: Vec<Rc<String>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn iter_test() {
        // Check iterating over data with a larger size than 8 bits.
        let vec_f32: Vec<Rc<f32>> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec_f32.clone()); // Convert into buffer
        for (i, val) in buf.iter::<Rc<f32>>().unwrap().enumerate() {
            assert_eq!(val, &vec_f32[i]);
        }

        // Check iterating over data with the same size.
        let vec_u8: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2, 4, 128, 32]
            .into_iter()
            .map(Rc::new)
            .collect();
        let buf = VecClone::from(vec_u8.clone()); // Convert into buffer
        for (i, val) in buf.iter::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }
    }

    #[test]
    fn large_sizes_clone() {
        for i in 100000..100010 {
            let vec: Vec<Rc<u8>> = vec![32u8; i].into_iter().map(Rc::new).collect();
            let buf = VecClone::from(vec.clone()); // Convert into buffer
            let nu_vec: Vec<Rc<u8>> = buf.into_vec().unwrap(); // Convert back into vec
            assert_eq!(vec, nu_vec);
        }
    }

    /// This test checks that an error is returned whenever the user tries to access data with the
    /// wrong type data.
    #[test]
    fn wrong_type_test() {
        let vec: Vec<Rc<f32>> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43]
            .into_iter()
            .map(Rc::new)
            .collect();
        let mut buf = VecClone::from(vec.clone()); // Convert into buffer
        assert_eq!(vec, buf.clone_into_vec::<Rc<f32>>().unwrap());

        assert!(buf.clone_into_vec::<Rc<f64>>().is_none());
        assert!(buf.as_slice::<Rc<f64>>().is_none());
        assert!(buf.as_mut_slice::<Rc<u8>>().is_none());
        assert!(buf.iter::<Rc<[f32; 3]>>().is_none());
        assert!(buf.get_ref::<Rc<i32>>(1).is_none());
        assert!(buf.get_mut::<Rc<i32>>(2).is_none());
    }

    /// Test pushing values and bytes to a buffer.
    #[test]
    fn push_test() {
        let mut vec_f32: Vec<Rc<f32>> =
            vec![1.0_f32, 23.0, 0.01].into_iter().map(Rc::new).collect();
        let mut buf = VecClone::from(vec_f32.clone()); // Convert into buffer
        for (i, val) in buf.iter::<Rc<f32>>().unwrap().enumerate() {
            assert_eq!(val, &vec_f32[i]);
        }

        vec_f32.push(Rc::new(42.0f32));
        buf.push(Rc::new(42.0f32)).unwrap(); // must provide explicit type

        for (i, val) in buf.iter::<Rc<f32>>().unwrap().enumerate() {
            assert_eq!(val, &vec_f32[i]);
        }

        vec_f32.push(Rc::new(11.43));
        buf.push(Rc::new(11.43f32)).unwrap();

        for (i, val) in buf.iter::<Rc<f32>>().unwrap().enumerate() {
            assert_eq!(val, &vec_f32[i]);
        }
    }

    /// Test appending to a buffer from another buffer.
    #[test]
    fn append_test() {
        let mut buf = VecClone::with_type::<Rc<f32>>(); // Create an empty buffer.

        let data: Vec<Rc<f32>> = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43]
            .into_iter()
            .map(Rc::new)
            .collect();
        // Append an ordianry vector of data.
        let mut other_buf = VecClone::from_vec(data.clone());
        buf.append(&mut other_buf);

        assert!(other_buf.is_empty());

        for (i, val) in buf.iter::<Rc<f32>>().unwrap().enumerate() {
            assert_eq!(val, &data[i]);
        }
    }
}
