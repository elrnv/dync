//! This crate defines a buffer data structure optimized to be written to and read from standard
//! `Vec`s. `DataBuffer` is particularly useful when dealing with data whose type is determined at
//! run time.  Note that data is stored in the underlying byte buffers in native endian form, thus
//! requesting typed data from a buffer on a platform with different endianness will produce
//! undefined behavior.

pub extern crate reinterpret;

#[cfg(feature = "numeric")]
extern crate num_traits;

use std::{
    any::{Any, TypeId}, mem::size_of, slice,
};

#[cfg(feature = "numeric")]
use std::fmt;

#[cfg(feature = "numeric")]
use num_traits::{NumCast, Zero, cast};

pub mod macros;

/// Buffer of plain old data (POD). The data is stored as an array of bytes (`Vec<u8>`).
/// `DataBuffer` keeps track of the type stored within via an explicit `TypeId` member. This allows
/// one to hide the type from the compiler and check it only when necessary. It is particularly
/// useful when the type of data is determined at runtime (e.g. when parsing numeric data).
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct DataBuffer {
    /// Raw data stored as an array of bytes.
    data: Vec<u8>,
    /// Number of type sized chunks in the buffer.
    length: usize,
    /// Type encoding for hiding the type of data from the compiler.
    type_id: TypeId,
}

impl DataBuffer {
    /// Construct an empty `DataBuffer`.
    #[inline]
    pub fn new() -> Self {
        DataBuffer {
            data: Vec::new(),
            length: 0,
            type_id: TypeId::of::<()>(),
        }
    }

    /// Construct an empty `DataBuffer` with a specific type.
    #[inline]
    pub fn with_type<T: Any>() -> Self {
        DataBuffer {
            data: Vec::new(),
            length: 0,
            type_id: TypeId::of::<T>(),
        }
    }

    /// Construct an empty `DataBuffer` with a capacity for a given number of typed elements. For
    /// setting byte capacity use `with_byte_capacity`.
    #[inline]
    pub fn with_capacity<T: Any>(n: usize) -> Self {
        DataBuffer {
            data: Vec::with_capacity(n*size_of::<T>()),
            length: 0,
            type_id: TypeId::of::<T>(),
        }
    }

    /// Construct an empty `DataBuffer` with a capacity for a given number of bytes.
    #[inline]
    pub fn with_byte_capacity(n: usize) -> Self {
        DataBuffer {
            data: Vec::with_capacity(n),
            length: 0,
            type_id: TypeId::of::<()>(),
        }
    }

    /// Construct a typed `DataBuffer` with a given size and filled with the specified default
    /// value.
    /// #  Examples
    /// ```
    /// # extern crate data_buffer as buf;
    /// # use buf::DataBuffer;
    /// # fn main() {
    /// let buf = DataBuffer::with_size(8, 42usize); // Create buffer
    /// let buf_vec: Vec<usize> = buf.into_vec().unwrap(); // Convert into `Vec`
    /// assert_eq!(buf_vec, vec![42usize; 8]);
    /// # }
    /// ```
    #[inline]
    pub fn with_size<T: Any + Clone>(n: usize, def: T) -> Self {
        let mut vec = Vec::with_capacity(n);
        vec.resize(n, def);
        Self::from_vec(vec)
    }

    /// Construct a `DataBuffer` from a given `Vec<T>` reusing the space already allocated by the
    /// given vector.
    /// #  Examples
    /// ```
    /// # extern crate data_buffer as buf;
    /// # use buf::DataBuffer;
    /// # fn main() {
    /// let vec = vec![1u8, 3, 4, 1, 2];
    /// let buf = DataBuffer::from_vec(vec.clone()); // Convert into buffer
    /// let nu_vec: Vec<u8> = buf.into_vec().unwrap(); // Convert back into `Vec`
    /// assert_eq!(vec, nu_vec);
    /// # }
    /// ```
    pub fn from_vec<T: Any>(mut vec: Vec<T>) -> Self {
        let length = vec.len();

        let data = {
            let len_in_bytes = length * size_of::<T>();
            let capacity_in_bytes = vec.capacity() * size_of::<T>();
            let vec_ptr = vec.as_mut_ptr() as *mut u8;

            unsafe {
                ::std::mem::forget(vec);
                Vec::from_raw_parts(vec_ptr, len_in_bytes, capacity_in_bytes)
            }
        };

        DataBuffer {
            data,
            length,
            type_id: TypeId::of::<T>(),
        }
    }

    /// Construct a `DataBuffer` from a given slice by cloning the data.
    #[inline]
    pub fn from_slice<T: Any + Clone>(slice: &[T]) -> Self {
        let mut vec = Vec::with_capacity(slice.len());
        vec.extend_from_slice(slice);
        Self::from_vec(vec)
    }

    /// Copy data from a given slice into the current buffer.
    #[inline]
    pub fn copy_from_slice<T: Any + Copy>(&mut self, slice: &[T]) -> &mut Self {
        let length = slice.len();
        let bins = length * size_of::<T>();
        let byte_slice = unsafe { slice::from_raw_parts(slice.as_ptr() as *const u8, bins) };
        self.data.resize(bins, 0);
        self.data.copy_from_slice(byte_slice);
        self.length = length;
        self.type_id = TypeId::of::<T>();
        self
    }

    /// Clear the data buffer and set length to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.length = 0;
        self.type_id = TypeId::of::<()>();
    }

    /// Fill the current buffer with copies of the given value. The size of the buffer is left
    /// unchanged. If the given type doesn't patch the internal type, `None` is returned, otherwise
    /// a mut reference to the modified buffer is returned.
    /// #  Examples
    /// ```
    /// # extern crate data_buffer as buf;
    /// # use buf::DataBuffer;
    /// # fn main() {
    /// let vec = vec![1u8, 3, 4, 1, 2];
    /// let mut buf = DataBuffer::from_vec(vec.clone()); // Convert into buffer
    /// buf.fill(0u8);
    /// assert_eq!(buf.into_vec::<u8>().unwrap(), vec![0u8, 0, 0, 0, 0]);
    /// # }
    /// ```
    #[inline]
    pub fn fill<T: Any + Clone>(&mut self, def: T) -> Option<&mut Self> {
        for v in self.iter_mut::<T>()? {
            *v = def.clone();
        }
        Some(self)
    }

    /// Add an element to this buffer. If the type of the given element coincides with the type
    /// stored by this buffer, then the modified buffer is returned via a mutable reference.
    /// Otherwise, `None` is returned.
    #[inline]
    pub fn push<T: Any + std::fmt::Debug>(&mut self, element: T) -> Option<&mut Self> {
        self.check_ref::<T>()?;
        let element_ref = &element;
        let element_byte_ptr = element_ref as *const T as *const u8;
        let element_byte_slice = unsafe {
            slice::from_raw_parts(element_byte_ptr, size_of::<T>())
        };
        self.push_bytes(element_byte_slice)
    }

    /// Check if the current buffer contains elements of the specified type. Returns `Some(self)`
    /// if the type matches and `None` otherwise.
    #[inline]
    pub fn check<T: Any>(self) -> Option<Self> {
        if TypeId::of::<T>() != self.type_id() { None } else { Some(self) }
    }

    /// Check if the current buffer contains elements of the specified type. Returns `None` if the
    /// check fails, otherwise a reference to self is returned.
    #[inline]
    pub fn check_ref<T: Any>(&self) -> Option<&Self> {
        if TypeId::of::<T>() != self.type_id() { None } else { Some(self) }
    }

    /// Check if the current buffer contains elements of the specified type. Same as `check_ref`
    /// but consumes and produces a mut reference to self.
    #[inline]
    pub fn check_mut<T: Any>(&mut self) -> Option<&mut Self> {
        if TypeId::of::<T>() != self.type_id() { None } else { Some(self) }
    }

    /*
     * Accessors
     */

    /// Get the `TypeId` of data stored within this buffer.
    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Get the number of elements stored in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Get the byte capacity of this buffer.
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Get the size of the element type.
    pub fn size_of_type(&self) -> usize {
        // Check that the length of the buffer is a multiple of the interpreted vector.
        // This must always be true because of how we build the `DataBuffer`.
        debug_assert_eq!( self.data.len() % self.length, 0 );
        self.data.len() / self.length
    }

    /// Return an iterator to a slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    /// # Examples
    /// ```
    /// # extern crate data_buffer as buf;
    /// # use buf::DataBuffer;
    /// # fn main() {
    /// let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
    /// let buf = DataBuffer::from(vec.clone()); // Convert into buffer
    /// for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
    ///     assert_eq!(val, vec[i]);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn iter<'a, T: Any + 'a>(&'a self) -> Option<slice::Iter<T>> {
        self.as_slice::<T>().map(|x| x.iter())
    }

    /// Return an iterator to a mutable slice representing typed data.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn iter_mut<'a, T: Any + 'a>(&'a mut self) -> Option<slice::IterMut<T>> {
        self.as_mut_slice::<T>().map(|x| x.iter_mut())
    }

    /// Append cloned items from this buffer to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise.
    #[inline]
    pub fn append_clone_to_vec<'a, T>(&self, vec: &'a mut Vec<T>) -> Option<&'a mut Vec<T>>
    where
        T: Any + Clone,
    {
        vec.extend_from_slice(self.as_slice()?);
        Some(vec)
    }

    /// Append copied items from this buffer to a given `Vec<T>`. Return the mutable reference
    /// `Some(vec)` if type matched the internal type and `None` otherwise. This may be faster than
    /// `append_clone_to_vec`.
    #[inline]
    pub fn append_copy_to_vec<'a, T>(&self, vec: &'a mut Vec<T>) -> Option<&'a mut Vec<T>>
    where
        T: Any + Copy,
    {
        vec.extend(self.iter()?);
        Some(vec)
    }

    /// Clones contents of `self` into the given `Vec`.
    #[inline]
    pub fn clone_into_vec<T: Any + Clone>(&self) -> Option<Vec<T>> {
        let mut vec = Vec::<T>::with_capacity(self.len());
        match self.append_clone_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }

    /// Copies contents of `self` into the given `Vec`.
    #[inline]
    pub fn copy_into_vec<T: Any + Copy>(&self) -> Option<Vec<T>> {
        let mut vec = Vec::<T>::with_capacity(self.len());
        match self.append_copy_to_vec(&mut vec) {
            Some(_) => Some(vec),
            None => None,
        }
    }

    /// An alternative to using the `Into` trait. This function helps the compiler
    /// determine the type `T` automatically.
    #[inline]
    pub fn into_vec<T: Any>(self) -> Option<Vec<T>> {
        self.check::<T>().map(|x| x.reinterpret_into_vec())
    }

    /// Convert this buffer into a typed slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_slice<T: Any>(&self) -> Option<&[T]> {
        let ptr = self.check_ref::<T>()?.data.as_ptr() as *const T;
        Some(unsafe { slice::from_raw_parts(ptr, self.len()) })
    }

    /// Convert this buffer into a typed mutable slice.
    /// Returs `None` if the given type `T` doesn't match the internal.
    #[inline]
    pub fn as_mut_slice<T: Any>(&mut self) -> Option<&mut [T]> {
        let ptr = self.check_mut::<T>()?.data.as_mut_ptr() as *mut T;
        Some(unsafe { slice::from_raw_parts_mut(ptr, self.len()) })
    }

    /// Get `i`'th element of the buffer by value.
    #[inline]
    pub fn get<T: Any + Copy>(&self, i: usize) -> Option<T> {
        assert!(i < self.len());
        let ptr = self.check_ref::<T>()?.data.as_ptr() as *const T;
        Some(unsafe { *ptr.offset(i as isize) })
    }

    /// Get a `const` reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_ref<T: Any>(&self, i: usize) -> Option<&T> {
        assert!(i < self.len());
        let ptr = self.check_ref::<T>()?.data.as_ptr() as *const T;
        Some(unsafe { &*ptr.offset(i as isize) })
    }

    /// Get a mutable reference to the `i`'th element of the buffer.
    #[inline]
    pub fn get_mut<T: Any>(&mut self, i: usize) -> Option<&mut T> {
        assert!(i < self.len());
        let ptr = self.check_mut::<T>()?.data.as_mut_ptr() as *mut T;
        Some(unsafe { &mut *ptr.offset(i as isize) })
    }

    /*
     * Advanced methods to probe buffer internals.
     */

    /// Get `i`'th element of the buffer by value without checking type.
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current buffer. See the implementation for details.
    #[inline]
    pub unsafe fn get_unchecked<T: Any + Copy>(&self, i: usize) -> T {
        let ptr = self.data.as_ptr() as *const T;
        *ptr.offset(i as isize)
    }

    /// Get a `const` reference to the `i`'th element of the buffer.
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current buffer. See the implementation for details.
    #[inline]
    pub unsafe fn get_unchecked_ref<T: Any>(&self, i: usize) -> &T {
        let ptr = self.data.as_ptr() as *const T;
        &*ptr.offset(i as isize)
    }

    /// Get a mutable reference to the `i`'th element of the buffer.
    /// This can be used to reinterpret the internal data as a different type. Note that if the
    /// size of the given type `T` doesn't match the size of the internal type, `i` will really
    /// index the `i`th `T` sized chunk in the current buffer. See the implementation for details.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: Any>(&mut self, i: usize) -> &mut T {
        let ptr = self.data.as_mut_ptr() as *mut T;
        &mut *ptr.offset(i as isize)
    }

    /// Move buffer data to a vector with a given type, reinterpreting the data type as
    /// required.
    #[inline]
    pub fn reinterpret_into_vec<T>(self) -> Vec<T> {
        reinterpret::reinterpret_vec(self.data)
    }

    /// Borrow buffer data and reinterpret it as a slice of a given type.
    #[inline]
    pub fn reinterpret_as_slice<T>(&self) -> &[T] {
        reinterpret::reinterpret_slice(self.data.as_slice())
    }

    /// Mutably borrow buffer data and reinterpret it as a mutable slice of a given type.
    #[inline]
    pub fn reinterpret_as_mut_slice<T>(&mut self) -> &mut [T] {
        reinterpret::reinterpret_mut_slice(self.data.as_mut_slice())
    }

    /// Borrow buffer data and iterate over reinterpreted underlying data.
    #[inline]
    pub fn reinterpret_iter<T>(&self) -> slice::Iter<T> {
        self.reinterpret_as_slice().iter()
    }

    /// Mutably borrow buffer data and mutably iterate over reinterpreted underlying data.
    #[inline]
    pub fn reinterpret_iter_mut<T>(&mut self) -> slice::IterMut<T> {
        self.reinterpret_as_mut_slice().iter_mut()
    }

    /// Peak at the internal representation of the data.
    #[inline]
    pub fn bytes_ref(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Get a mutable reference to the internal data representation.
    #[inline]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.data.as_mut_slice()
    }

    /// Iterate over chunks type sized chunks of bytes without interpreting them. This avoids
    /// needing to know what type data you're dealing with. This type of iterator is useful for
    /// transferring data from one place to another for a generic buffer.
    #[inline]
    pub fn byte_chunks<'a>(&'a self) -> impl Iterator<Item=&'a [u8]> + 'a {
        let chunk_size = self.size_of_type();
        self.data.chunks(chunk_size)
    }

    /// Mutably iterate over chunks type sized chunks of bytes without interpreting them. This
    /// avoids needing to know what type data you're dealing with. This type of iterator is useful
    /// for transferring data from one place to another for a generic buffer, or modifying the
    /// underlying untyped bytes (e.g. bit twiddling).
    #[inline]
    pub fn byte_chunks_mut<'a>(&'a mut self) -> impl Iterator<Item=&'a mut [u8]> + 'a {
        let chunk_size = self.size_of_type();
        self.data.chunks_mut(chunk_size)
    }

    /// Add bytes to this buffer. If the size of the given slice coincides with the number of bytes
    /// occupied by the underlying element type, then these bytes are added to the underlying data
    /// buffer and a mutable reference to the buffer is returned.
    /// Otherwise, `None` is returned, and the buffer remains unmodified.
    #[inline]
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Option<&mut Self> {
        if bytes.len() == self.size_of_type() { 
            self.data.extend_from_slice(bytes);
            self.length += 1;
            Some(self)
        } else {
            None
        }
    }

    /// Add bytes to this buffer. If the size of the given slice is a multiple of the number of bytes
    /// occupied by the underlying element type, then these bytes are added to the underlying data
    /// buffer and a mutable reference to the buffer is returned.
    /// Otherwise, `None` is returned and the buffer is unmodified.
    #[inline]
    pub fn extend_bytes(&mut self, bytes: &[u8]) -> Option<&mut Self> {
        let size_of_type = self.size_of_type();
        if bytes.len() % size_of_type == 0 { 
            self.data.extend_from_slice(bytes);
            self.length += bytes.len() / size_of_type;
            Some(self)
        } else {
            None
        }
    }

    /*
     * Methods specific to buffers storing numeric data
     */

    #[cfg(feature = "numeric")]
    /// Cast a numeric `DataBuffer` into the given output `Vec` type.
    pub fn cast_into_vec<T>(self) -> Vec<T>
        where T: Any + Copy + NumCast + Zero
    {
        // Helper function (generic on the input) to conver the given DataBuffer into Vec.
        fn convert_into_vec<I,O>(buf: DataBuffer) -> Vec<O>
            where I: Any + NumCast,
                  O: Any + Copy + NumCast + Zero
        {
            debug_assert_eq!(buf.type_id(), TypeId::of::<I>()); // Check invariant.
            buf.reinterpret_into_vec()
               .into_iter()
               .map(|elem: I| cast(elem).unwrap_or(O::zero())).collect()
        }
        call_numeric_buffer_fn!( convert_into_vec::<_,T>(self) or { Vec::new() } )
    }

    #[cfg(feature = "numeric")]
    /// Display the contents of this buffer reinterpreted in the given type.
    fn reinterpret_display<T: Any + fmt::Display>(&self, f: &mut fmt::Formatter) {
        debug_assert_eq!(self.type_id(), TypeId::of::<T>()); // Check invariant.
        for item in self.reinterpret_iter::<T>() {
            write!(f, "{} ", item)
                .expect("Error occurred while writing an DataBuffer.");
        }
    }

}

/// Convert a `Vec<T>` to a `DataBuffer`.
impl<T> From<Vec<T>> for DataBuffer
where
    T: Any,
{
    fn from(vec: Vec<T>) -> DataBuffer {
        DataBuffer::from_vec(vec)
    }
}

/// Convert a `&[T]` to a `DataBuffer`.
impl<'a, T> From<&'a [T]> for DataBuffer
where
    T: Any + Clone,
{
    fn from(slice: &'a [T]) -> DataBuffer {
        DataBuffer::from_slice(slice)
    }
}

/// Convert a `DataBuffer` to a `Option<Vec<T>>`.
impl<T> Into<Option<Vec<T>>> for DataBuffer
where
    T: Any + Clone,
{
    fn into(self) -> Option<Vec<T>> {
        self.into_vec()
    }
}

#[cfg(feature = "numeric")]
/// Implement pretty printing of numeric `DataBuffer` data.
impl fmt::Display for DataBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        call_numeric_buffer_fn!( self.reinterpret_display::<_>(f) or {
            println!("Unknown DataBuffer type for pretty printing.");
        } );
        write!(f, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test various ways to create a data buffer.
    #[test]
    fn initialization_test() {
        // Empty untyped buffer.
        let a = DataBuffer::new();
        assert_eq!(a.len(), 0);
        assert_eq!(a.bytes_ref().len(), 0);
        assert_eq!(a.type_id(), TypeId::of::<()>());

        // Empty typed buffer.
        let a = DataBuffer::with_type::<f32>();
        assert_eq!(a.len(), 0);
        assert_eq!(a.bytes_ref().len(), 0);
        assert_eq!(a.type_id(), TypeId::of::<f32>());

        // Empty typed buffer with a given capacity.
        let a = DataBuffer::with_capacity::<f32>(4);
        assert_eq!(a.len(), 0);
        assert_eq!(a.bytes_ref().len(), 0);
        assert_eq!(a.byte_capacity(), 4*size_of::<f32>());
        assert_eq!(a.type_id(), TypeId::of::<f32>());

        // Empty untyped buffer with a given byte capacity.
        let a = DataBuffer::with_byte_capacity(4);
        assert_eq!(a.len(), 0);
        assert_eq!(a.bytes_ref().len(), 0);
        assert_eq!(a.byte_capacity(), 4);
        assert_eq!(a.type_id(), TypeId::of::<()>());
    }

    #[test]
    fn data_integrity_u8_test() {
        let vec = vec![1u8, 3, 4, 1, 2];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<u8> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1u8, 3, 4, 1, 2, 52, 1, 3, 41, 23, 2];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<u8> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i16_test() {
        let vec = vec![1i16, -3, 1002, -231, 32];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i16> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1i16, -3, 1002, -231, 32, 42, -123, 4];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i16> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_i32_test() {
        let vec = vec![1i32, -3, 1002, -231, 32];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1i32, -3, 1002, -231, 32, 42, -123];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<i32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_f32_test() {
        let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43, 2e-1];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn data_integrity_f64_test() {
        let vec = vec![1f64, -3.0, 10.02, -23.1, 32e-1];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec = vec![1f64, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.copy_into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[cfg(feature = "numeric")]
    #[test]
    fn convert_float_test() {
        let vecf64 = vec![1f64, -3.0, 10.02, -23.1, 32e-1];
        let buf = DataBuffer::from(vecf64.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.cast_into_vec(); // Convert back into vec
        let vecf32 = vec![1f32, -3.0, 10.02, -23.1, 32e-1];
        assert_eq!(vecf32, nu_vec);

        let buf = DataBuffer::from(vecf32.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.cast_into_vec(); // Convert back into vec
        for (&a, &b) in vecf64.iter().zip(nu_vec.iter()) {
            assert!((a - b).abs() < 1e-6f64*f64::max(a,b).abs());
        }

        let vecf64 = vec![1f64, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        let buf = DataBuffer::from(vecf64.clone()); // Convert into buffer
        let nu_vec: Vec<f32> = buf.cast_into_vec(); // Convert back into vec
        let vecf32 = vec![1f32, -3.1, 100.2, -2.31, 3.2, 4e2, -1e23];
        assert_eq!(vecf32, nu_vec);
        let buf = DataBuffer::from(vecf32.clone()); // Convert into buffer
        let nu_vec: Vec<f64> = buf.cast_into_vec(); // Convert back into vec
        for (&a, &b) in vecf64.iter().zip(nu_vec.iter()) {
            assert!((a - b).abs() < 1e-6*f64::max(a,b).abs());
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Foo {
        a: u8,
        b: i64,
        c: f32,
    }

    #[test]
    fn from_empty_vec_test() {
        let vec: Vec<u32> = Vec::new();
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<u32> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<String> = Vec::new();
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        let nu_vec: Vec<String> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);

        let vec: Vec<Foo> = Vec::new();
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
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
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        assert_eq!(f1, buf.get_ref::<Foo>(0).unwrap().clone());
        assert_eq!(f2, buf.get_ref::<Foo>(1).unwrap().clone());
        let nu_vec: Vec<Foo> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn from_strings_test() {
        let vec = vec![
            String::from("hi"),
            String::from("hello"),
            String::from("goodbye"),
            String::from("bye"),
            String::from("supercalifragilisticexpialidocious"),
            String::from("42"),
        ];
        let buf = DataBuffer::from(vec.clone()); // Convert into buffer
        assert_eq!("hi", buf.get_ref::<String>(0).unwrap());
        assert_eq!("hello", buf.get_ref::<String>(1).unwrap());
        assert_eq!("goodbye", buf.get_ref::<String>(2).unwrap());
        let nu_vec: Vec<String> = buf.into_vec().unwrap(); // Convert back into vec
        assert_eq!(vec, nu_vec);
    }

    #[test]
    fn iter_test() {
        // Check iterating over data with a larger size than 8 bits.
        let vec_f32 = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let buf = DataBuffer::from(vec_f32.clone()); // Convert into buffer
        for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        // Check iterating over data with the same size.
        let vec_u8 = vec![1u8, 3, 4, 1, 2, 4, 128, 32];
        let buf = DataBuffer::from(vec_u8.clone()); // Convert into buffer
        for (i, &val) in buf.iter::<u8>().unwrap().enumerate() {
            assert_eq!(val, vec_u8[i]);
        }

        // TODO: feature gate these two tests for little endian platforms.
        // Check iterating over data with a larger size than input.
        let vec_u32 = vec![17_040_129u32, 545_260_546]; // little endian
        let buf = DataBuffer::from(vec_u8.clone()); // Convert into buffer
        for (i, &val) in buf.reinterpret_iter::<u32>().enumerate() {
            assert_eq!(val, vec_u32[i]);
        }

        // Check iterating over data with a smaller size than input
        let mut buf2 = DataBuffer::from(vec_u32); // Convert into buffer
        for (i, &val) in buf2.reinterpret_iter::<u8>().enumerate() {
            assert_eq!(val, vec_u8[i]);
        }

        // Check mut iterator
        buf2.reinterpret_iter_mut::<u8>().for_each(|val| *val += 1);

        let u8_check_vec = vec![2u8, 4, 5, 2, 3, 5, 129, 33];
        assert_eq!(buf2.reinterpret_into_vec::<u8>(), u8_check_vec);
    }

    #[test]
    fn large_sizes_test() {
        for i in 1000000..1000010 {
            let vec = vec![32u8; i];
            let buf = DataBuffer::from(vec.clone()); // Convert into buffer
            let nu_vec: Vec<u8> = buf.into_vec().unwrap(); // Convert back into vec
            assert_eq!(vec, nu_vec);
        }
    }

    /// This test checks that an error is returned whenever the user tries to access data with the
    /// wrong type data.
    #[test]
    fn wrong_type_test() {
        let vec = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let mut buf = DataBuffer::from(vec.clone()); // Convert into buffer
        assert_eq!(vec, buf.clone_into_vec().unwrap());

        assert!(buf.copy_into_vec::<f64>().is_none());
        assert!(buf.as_slice::<f64>().is_none());
        assert!(buf.as_mut_slice::<u8>().is_none());
        assert!(buf.iter::<[f32; 3]>().is_none());
        assert!(buf.get::<i32>(0).is_none());
        assert!(buf.get_ref::<i32>(1).is_none());
        assert!(buf.get_mut::<i32>(2).is_none());
    }

    /// Test iterating over chunks of data without having to interpret them.
    #[test]
    fn byte_chunks_test() {
        let vec_f32 = vec![1.0_f32, 23.0, 0.01, 42.0, 11.43];
        let buf = DataBuffer::from(vec_f32.clone()); // Convert into buffer

        for (i, val) in buf.byte_chunks().enumerate() {
            assert_eq!(reinterpret::reinterpret_slice::<u8, f32>(val)[0], vec_f32[i]);
        }
    }

    /// Test pushing values and bytes to a buffer.
    #[test]
    fn push_test() {
        let mut vec_f32 = vec![1.0_f32, 23.0, 0.01];
        let mut buf = DataBuffer::from(vec_f32.clone()); // Convert into buffer
        for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        vec_f32.push(42.0f32);
        buf.push(42.0f32).unwrap(); // must provide explicit type

        for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        vec_f32.push(11.43);
        buf.push(11.43f32).unwrap();

        for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        // Zero float is always represented by four zero bytes in IEEE format.
        vec_f32.push(0.0);
        vec_f32.push(0.0);
        buf.extend_bytes(&[0,0,0,0, 0,0,0,0]).unwrap();

        for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }

        vec_f32.push(0.0);
        buf.push_bytes(&[0,0,0,0]).unwrap();

        for (i, &val) in buf.iter::<f32>().unwrap().enumerate() {
            assert_eq!(val, vec_f32[i]);
        }
    }
}
