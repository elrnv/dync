//! Utility macros for code generation.

#![macro_use]

/// Applies `$fn` to an `VecCopy` mapping valid numeric data types by corresponding generic
/// parameters.  For example, passing an `VecCopy` containing data of type `u8` will cause this
/// macro to call `$fn` with type parameter `u8` like `$fn::<u8>(buffer)`.
/// # Examples
/// ```rust
/// # #[macro_use] extern crate dync;
/// # use std::fmt;
/// # use std::any::Any;
/// # use dync::VecCopy;
/// // Implement pretty printing of a `VecCopy` derivative for numeric buffers.
/// struct MyBuffer(VecCopy);
/// impl fmt::Display for MyBuffer {
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         unsafe fn display_buf<T: Copy + Any + fmt::Display>(buf: &VecCopy, f: &mut fmt::Formatter) {
///             for item in buf.iter::<T>().unwrap() {
///                 write!(f, "{} ", item)
///                     .expect("Error occurred while writing MyBuffer.");
///             }
///         }
///         call_numeric_buffer_fn!( display_buf::<_>(&self.0, f) or {});
///         write!(f, "")
///     }
/// }
/// ```
#[macro_export]
macro_rules! call_numeric_buffer_fn {
    ($fn:ident ::<_,$($params:ident),*>( $data:expr, $($args:expr),* ) or $err:block ) => {
        {
            let buf = $data;
            unsafe {
                match buf.element_type_id() {
                    x if x == ::std::any::TypeId::of::<u8>() =>  $fn::<u8,$($params),*> (buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<i8>() =>  $fn::<i8,$($params),*> (buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<u16>() => $fn::<u16,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<i16>() => $fn::<i16,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<u32>() => $fn::<u32,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<i32>() => $fn::<i32,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<u64>() => $fn::<u64,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<i64>() => $fn::<i64,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<f32>() => $fn::<f32,$($params),*>(buf, $($args),*),
                    x if x == ::std::any::TypeId::of::<f64>() => $fn::<f64,$($params),*>(buf, $($args),*),
                    _ => $err,
                }
            }
        }
    };
    // Same thing as above but with one parameter argument.
    ($fn:ident ::<_>( $($args:expr),* ) or $err:block ) => {
        call_numeric_buffer_fn!($fn ::<_,>( $($args),* ) or $err )
    };
    // Same thing as above but with one function argument.
    ($fn:ident ::<_,$($params:ident),*>( $data:expr ) or $err:block ) => {
        call_numeric_buffer_fn!($fn ::<_,$($params),*>( $data, ) or $err )
    };
    // Using method synax for member functions if any.
    ($data:ident . $fn:ident ::<_,$($params:ident),*>( $($args:expr),* ) or $err:block ) => {
        {
            let buf = $data;
            unsafe {
                match buf.element_type_id() {
                    x if x == ::std::any::TypeId::of::<u8>() =>  buf.$fn::<u8,$($params),*> ($($args),*),
                    x if x == ::std::any::TypeId::of::<i8>() =>  buf.$fn::<i8,$($params),*> ($($args),*),
                    x if x == ::std::any::TypeId::of::<u16>() => buf.$fn::<u16,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<i16>() => buf.$fn::<i16,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<u32>() => buf.$fn::<u32,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<i32>() => buf.$fn::<i32,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<u64>() => buf.$fn::<u64,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<i64>() => buf.$fn::<i64,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<f32>() => buf.$fn::<f32,$($params),*>($($args),*),
                    x if x == ::std::any::TypeId::of::<f64>() => buf.$fn::<f64,$($params),*>($($args),*),
                    _ => $err,
                }
            }
        }
    };
    // Same as above but with one parameter argument.
    ($data:ident . $fn:ident ::<_>( $($args:expr),* ) or $err:block ) => {
        call_numeric_buffer_fn!($data . $fn ::<_,>( $($args),* ) or $err )
    };
}
