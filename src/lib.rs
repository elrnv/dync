//! This crate defines a buffer data structure optimized to be written to and read from standard
//! `Vec`s.
//!
//! [`VecCopy`] is particularly useful when dealing with plain data whose type is determined at
//! run time.  Note that data is stored in the underlying byte buffers in native endian form, thus
//! requesting typed data from a buffer on a platform with different endianness is unsafe.
//!
//! # Caveats
//!
//! [`VecCopy`] doesn't support zero-sized types.
//!
//! [`VecCopy`]: struct.VecCopy

pub mod macros;
mod bytes;
pub mod traits;
#[macro_use]
mod value;
mod slice_copy;
mod slice_dyn;
mod vec_copy;
mod vec_dyn;

pub use dync_derive::dync_trait;
pub use slice_copy::*;
pub use slice_dyn::*;
pub use value::*;
pub use vec_copy::*;
pub use vec_dyn::*;
