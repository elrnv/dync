#![allow(dead_code)]
use std::{
    any::{Any, TypeId},
    mem::ManuallyDrop,
    slice,
    sync::Arc,
    fmt,
};

use crate::bytes::*;
use crate::value::*;
use crate::traits::*;
use crate::VecCopy;

pub trait Elem: Any + Bytes + DropBytes {}
impl<T> Elem for T where T: Any + Bytes + DropBytes {}

/// This container is a WIP, not to be used in production.
#[derive(Hash)]
pub struct VecDyn<V> {
    data: ManuallyDrop<VecCopy>,
    vtable: Arc<(DropFn, V)>,
}

impl<V: HasDebug> fmt::Debug for VecDyn<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter_value_ref()).finish()
    }
}

impl<V: HasClone> Clone for VecDyn<V> {
    fn clone(&self) -> Self {
        let data_clone = |bytes: &[u8]| {
            let mut new_data = bytes.to_vec();
            self.data
                .byte_chunks()
                .zip(new_data.chunks_exact_mut(self.data.element_size()))
                .for_each(|(src, dst)| unsafe { self.vtable.1.clone_from_fn()(dst, src) });
            new_data
        };
        VecDyn {
            data: ManuallyDrop::new(self.data.clone_with(data_clone)),
            vtable: Arc::clone(&self.vtable),
        }
    }
}

impl<V> Drop for VecDyn<V> {
    fn drop(&mut self) {
        unsafe {
            for elem_bytes in self.data.byte_chunks_mut() {
                self.vtable.drop_fn().0(elem_bytes);
            }
        }
    }
}

impl<V> VecDyn<V> {
    /// Construct an empty vector with a specific pointed-to element type.
    #[inline]
    pub fn with_type<T: Elem + Dyn<VTable = V>>() -> Self {
        VecDyn {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::with_type_non_copy::<T>() }),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
        }
    }

    /// Construct a vector with the same type as the given vector without copying its data.
    #[inline]
    pub fn with_type_from(other: &VecDyn<V>) -> Self {
        VecDyn {
            data: ManuallyDrop::new(VecCopy::with_type_from(&other.data)),
            vtable: Arc::clone(&other.vtable),
        }
    }

    /// Construct an empty vector with a capacity for a given number of typed pointed-to elements.
    #[inline]
    pub fn with_capacity<T: Elem + Dyn<VTable = V>>(n: usize) -> Self {
        VecDyn {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new( unsafe { VecCopy::with_capacity_non_copy::<T>(n) }),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
        }
    }

    /// Construct a `VecDyn` from a given `Vec` reusing the space already allocated by the given
    /// vector.
    pub fn from_vec<T: Elem + Dyn<VTable = V>>(vec: Vec<T>) -> Self {
        VecDyn {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_vec_non_copy(vec) }),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
        }
    }

    /// Clear the data buffer without destroying its type information.
    #[inline]
    pub fn clear(&mut self) {
        // Drop all elements manually.
        unsafe {
            for bytes in self.data.byte_chunks_mut() {
                self.vtable.drop_fn().0(bytes);
            }
        }
        self.data.data.clear();
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
    pub fn append(&mut self, buf: &mut VecDyn<V>) -> Option<&mut Self> {
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
    pub fn value_ref(&self, i: usize) -> DynValueRef<V> {
        debug_assert!(i < self.len());
        // This call is safe since our buffer guarantees that the given bytes have the
        // corresponding TypeId.
        unsafe { DynValueRef::from_raw_parts(self.data.get_bytes(i), self.element_type_id(), Arc::clone(&self.vtable)) }
    }

    /// Return an iterator over untyped value references stored in this buffer.
    ///
    /// In contrast to `iter`, this function defers downcasting on a per element basis.
    /// As a result, this type of iteration is typically less efficient if a typed value is
    /// needed for each element.
    #[inline]
    pub fn iter_value_ref<'a>(&'a self) -> impl Iterator<Item = DynValueRef<'a, V>> + 'a {
        let &Self { ref data, ref vtable } = self;
        let VecCopy {
            data,
            element_size,
            element_type_id,
        } = &**data;
        data.chunks_exact(*element_size)
            .map(move |bytes| unsafe { DynValueRef::from_raw_parts(bytes, *element_type_id, Arc::clone(vtable)) })
    }
}

// Additional functionality of VecDyns that implement Clone.
impl<V: HasClone> VecDyn<V> {
    /// Construct a typed `DataBuffer` with a given size and filled with the specified default
    /// value.
    #[inline]
    pub fn with_size<T: Elem + Clone + Dyn<VTable = V>>(n: usize, def: T) -> Self {
        VecDyn {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_vec_non_copy(vec![def; n]) }),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
        }
    }

    /// Construct a buffer from a given slice by cloning the data.
    #[inline]
    pub fn from_slice<T: Elem + Clone + Dyn<VTable = V>>(slice: &[T]) -> Self {
        VecDyn {
            // This is safe because we are handling the additional processing needed
            // by `Clone` types in this container.
            data: ManuallyDrop::new(unsafe { VecCopy::from_slice_non_copy::<T>(slice) }),
            vtable: Arc::new((DropFn(T::drop_bytes), T::build_vtable()))
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
    pub fn resize<T: Elem + Clone>(&mut self, new_len: usize, value: T) -> Option<&mut Self> {
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
                    self.vtable.drop_fn().0(bytes);
                }
            }
            // Truncate data
            self.data.data.resize(new_len * size_t, 0);
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
        for v in self.iter_mut::<T>()? {
            *v = def.clone();
        }
        Some(self)
    }

    /// Append cloned items from this buffer to a given `Vec`.
    ///
    /// Return the mutable reference `Some(vec)` if type matched the internal type and
    /// `None` otherwise.
    #[inline]
    pub fn append_cloned_to_vec<'a, T: Elem + Clone>(&self, vec: &'a mut Vec<T>) -> Option<&'a mut Vec<T>> {
        let slice = self.as_slice()?;
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

    /// Get a mutable reference to a value stored in this container at index `i`.
    #[inline]
    pub fn value_mut(&mut self, i: usize) -> DynValueMut<'_, V> {
        debug_assert!(i < self.len());
        let Self {
            data,
            vtable,
        } = self;
        let type_id = data.element_type_id();
        // Safety is guaranteed here by the value API.
        unsafe {
            DynValueMut::from_raw_parts(data.get_bytes_mut(i), type_id, Arc::clone(vtable))
        }
    }

    /// Return an iterator over mutable untyped value references stored in this buffer.
    ///
    /// In contrast to `iter_mut`, this function defers downcasting on a per element basis.  As a
    /// result, this type of iteration is typically less efficient if a typed value is needed
    /// for each element.
    #[inline]
    pub fn iter_value_mut<'a>(&'a mut self) -> impl Iterator<Item = DynValueMut<'a, V>> + 'a {
        let &mut Self {
            ref mut data,
            ref mut vtable,
        } = self;
        let VecCopy {
            data,
            element_size,
            element_type_id,
        } = &mut **data;
        data.chunks_exact_mut(*element_size)
            .map(move |bytes| unsafe {
                DynValueMut::from_raw_parts(bytes, *element_type_id, Arc::clone(vtable))
            })
    }
}

/// Convert a `Vec` to a buffer.
impl<T: Elem + Dyn> From<Vec<T>> for VecDyn<T::VTable> {
    #[inline]
    fn from(vec: Vec<T>) -> VecDyn<T::VTable> {
        VecDyn::from_vec(vec)
    }
}

/// Convert a slice to a `VecDyn`.
impl<'a, T> From<&'a [T]> for VecDyn<T::VTable>
where T: Elem + Clone + Dyn,
      T::VTable: HasClone,
{
    #[inline]
    fn from(slice: &'a [T]) -> VecDyn<T::VTable> {
        VecDyn::from_slice(slice)
    }
}

/// Convert a buffer to a `Vec` with an option to fail.
impl<T: Elem, V> Into<Option<Vec<T>>> for VecDyn<V> {
    #[inline]
    fn into(self) -> Option<Vec<T>> {
        self.into_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use std::rc::Rc;
    use dyn_derive::dyn_trait;

    #[dyn_trait(suffix = "VTable", dyn_crate_name = "crate")]
    pub trait AllTrait: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug { }
    impl<T> AllTrait for T where T: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug { }

    #[test]
    fn clone_from_test() {
        //use std::collections::HashSet;
        //use std::rc::Rc;

        //// Let's create a collection of `Rc`s.
        //let vec_rc: Vec<_> = vec![1, 23, 2, 42, 23, 1, 135346534653]
        //    .into_iter()
        //    .map(Rc::new)
        //    .collect();
        //let buf = VecDyn::from(vec_rc.clone()); // Clone into VecDyn

        //// Construct a hashset of unique values from the VecDyn.
        //let mut hashset: HashSet<RcDynValue<AllTraitVTable>> = HashSet::new();

        //for rc in buf.iter_value_ref().take(4) {
        //    assert!(hashset.insert(Rc::clone(rc).into()));
        //}

        //assert!(!hashset.insert(Rc::clone(&vec_rc[4]).into()));
        //assert!(!hashset.insert(Rc::clone(&vec_rc[5]).into()));

        //assert_eq!(hashset.len(), 4);
        //assert!(hashset.contains(&Rc::new(1).into()));
        //assert!(hashset.contains(&Rc::new(23).into()));
        //assert!(hashset.contains(&Rc::new(2).into()));
        //assert!(hashset.contains(&Rc::new(42).into()));
        //assert!(!hashset.contains(&Rc::new(42.0).into()));
    }

    #[test]
    fn iter_value_ref() {
        use std::rc::Rc;
        let vec: Vec<_> = vec![1, 23, 2, 42, 11]
            .into_iter()
            .map(Rc::new)
            .collect();
        {
            let buf = VecDyn::from(vec.clone()); // Convert into buffer
            let orig = Rc::new(100);
            let mut rc = Rc::clone(&orig);
            assert_eq!(Rc::strong_count(&rc), 2);
            for val in buf.iter_value_ref() {
                DynValueMut::new(&mut rc).clone_from(val);
            }
            assert_eq!(Rc::strong_count(&orig), 1);
            assert_eq!(Rc::strong_count(&rc), 3);
            assert_eq!(Rc::strong_count(&vec[4]), 3);
            assert!(vec.iter().take(4).all(|x| Rc::strong_count(x) == 2));
            assert_eq!(rc, Rc::new(11));
        }
        assert!(vec.iter().all(|x| Rc::strong_count(x) == 1));
    }

    /// Test various ways to create a `VecDyn`.
    #[test]
    fn initialization_test() {
        // Empty typed buffer.
        let a = VecDyn::with_type::<Rc<u8>>();
        assert_eq!(a.len(), 0);
        assert_eq!(a.element_type_id(), TypeId::of::<Rc<u8>>());
        assert_eq!(a.byte_capacity(), 0); // Ensure nothing is allocated.

        // Empty buffer typed by the given type id.
        let b = VecDyn::with_type_from(&a);
        assert_eq!(b.len(), 0);
        assert_eq!(b.element_type_id(), TypeId::of::<Rc<u8>>());
        assert_eq!(a.byte_capacity(), 0); // Ensure nothing is allocated.

        // Empty typed buffer with a given capacity.
        let a = VecDyn::with_capacity::<Rc<u8>>(4);
        assert_eq!(a.len(), 0);
        assert_eq!(a.byte_capacity(), 4 * size_of::<Rc<u8>>());
        assert_eq!(a.element_type_id(), TypeId::of::<Rc<u8>>());
    }

    /// Test resizing a buffer.
    #[test]
    fn resize() {
        let mut a = VecDyn::with_type::<Rc<u8>>();

        // Increase the size of a.
        a.resize(3, Rc::new(1u8)).expect("Failed to resize VecDyn up by 3 elements");

        assert_eq!(a.len(), 3);
        for i in 0..3 {
            assert_eq!(a.get_ref::<Rc<u8>>(i).unwrap(), &Rc::new(1));
        }

        // Truncate a.
        a.resize(2, Rc::new(1u8)).expect("Failed to resize VecDyn down to 2 elements");

        assert_eq!(a.len(), 2);
        for i in 0..2 {
            assert_eq!(a.get_ref::<Rc<u8>>(i).unwrap(), &Rc::new(1));
        }
    }

    #[test]
    fn data_integrity_u8_test() {
        let vec: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u8>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2, 52, 1, 3, 41, 23, 2].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u8>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i16_test() {
        let vec: Vec<Rc<i16>> = vec![1i16, -3, 1002, -231, 32].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i16>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<i16>> = vec![1i16, -3, 1002, -231, 32, 42, -123, 4].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i16>> = buf.clone_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i32_test() {
        let vec: Vec<Rc<i32>> = vec![1i32, -3, 1002, -231, 32].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<i32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<i32>> = vec![1i32, -3, 1002, -231, 32, 42, -123].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
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
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<u32>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<String>> = Vec::new();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<String>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Rc<Foo>> = Vec::new();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<Rc<Foo>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn from_struct_test() {
        let f1 = Foo {
            a: 3,
            b: -32,
        };
        let f2 = Foo {
            a: 33,
            b: -3342432412,
        };
        let vec: Vec<Rc<Foo>> = vec![Rc::new(f1.clone()), Rc::new(f2.clone())];
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
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
        ].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec.clone()); // Convert into buffer
        assert_eq!(&Rc::new("hi".to_string()), buf.get_ref::<Rc<String>>(0).unwrap());
        assert_eq!(&Rc::new("hello".to_string()), buf.get_ref::<Rc<String>>(1).unwrap());
        assert_eq!(&Rc::new("goodbye".to_string()), buf.get_ref::<Rc<String>>(2).unwrap());
        let nu_vec: Vec<Rc<String>> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn iter_test() {
        let vec_u8: Vec<Rc<u8>> = vec![1u8, 3, 4, 1, 2, 4, 128, 32].into_iter().map(Rc::new).collect();
        let buf = VecDyn::from(vec_u8.clone()); // Convert into buffer
        for (i, val) in buf.iter::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }
    }

    #[test]
    fn large_sizes_clone() {
        for i in 100000..100010 {
            let vec: Vec<Rc<u8>> = vec![32u8; i].into_iter().map(Rc::new).collect();
            let buf = VecDyn::from(vec.clone()); // Convert into buffer
            let nu_vec: Vec<Rc<u8>> = buf.into_vec().unwrap(); // Convert back into vec
            assert_eq!(vec, nu_vec);
        }
    }

    /// This test checks that an error is returned whenever the user tries to access data with the
    /// wrong type data.
    #[test]
    fn wrong_type_test() {
        let vec: Vec<Rc<u8>> = vec![1, 23, 2, 42, 11].into_iter().map(Rc::new).collect();
        let mut buf = VecDyn::from(vec.clone()); // Convert into buffer
        assert_eq!(vec, buf.clone_into_vec::<Rc<u8>>().unwrap());

        assert!(buf.clone_into_vec::<Rc<f64>>().is_none());
        assert!(buf.as_slice::<Rc<f64>>().is_none());
        assert!(buf.iter::<Rc<[u8; 3]>>().is_none());
        assert!(buf.get_ref::<Rc<i32>>(1).is_none());
        assert!(buf.get_mut::<Rc<i32>>(2).is_none());
    }

    /// Test pushing values and bytes to a buffer.
    #[test]
    fn push_test() {
        let mut vec_u8: Vec<Rc<u8>> = vec![1u8, 23, 2].into_iter().map(Rc::new).collect();
        let mut buf = VecDyn::from(vec_u8.clone()); // Convert into buffer
        for (i, val) in buf.iter::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }

        vec_u8.push(Rc::new(42u8));
        buf.push(Rc::new(42u8)).unwrap(); // must provide explicit type

        for (i, val) in buf.iter::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }

        vec_u8.push(Rc::new(11u8));
        buf.push(Rc::new(11u8)).unwrap();

        for (i, val) in buf.iter::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &vec_u8[i]);
        }
    }

    /// Test appending to a buffer from another buffer.
    #[test]
    fn append_test() {
        let mut buf = VecDyn::with_type::<Rc<u8>>(); // Create an empty buffer.

        let data: Vec<Rc<u8>> = vec![1, 23, 2, 42, 11].into_iter().map(Rc::new).collect();
        // Append an ordianry vector of data.
        let mut other_buf = VecDyn::from_vec(data.clone());
        buf.append(&mut other_buf);

        assert!(other_buf.is_empty());

        for (i, val) in buf.iter::<Rc<u8>>().unwrap().enumerate() {
            assert_eq!(val, &data[i]);
        }
    }
}
