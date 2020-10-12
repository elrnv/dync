# `dync`

An efficient alternative to `dyn Trait` for containerized types.

[![On crates.io](https://img.shields.io/crates/v/dync.svg)](https://crates.io/crates/dync)
[![On docs.rs](https://docs.rs/dync/badge.svg)](https://docs.rs/dync/)
[![Build status](https://github.com/elrnv/dync/workflows/CI/badge.svg)](https://github.com/elrnv/dync/actions?query=workflow%3ACI)

# Overview

This crate aims to fill the gap in Rust's dynamic traits system by exposing the control over dynamic
virtual function tables to the user in a safe API. This library is in prototype stage and is not
recommended for production use.

Currently only untyped `Vec`s and `Slice`s are implemented. Complementary `Value` types
are provided for compatibility with traditional, typed containers like `VecDeque`s and `HashMap`s.

Notably `dync` introduces the following types:
 - `VecCopy<V>` - a homogeneous collection of `Copy` types generic over the virtual function table, 
 - `VecDrop<V>` - a homogeneous collection generic over the virtual function table, 
 - `SliceCopy<V>` (and `SliceCopyMut<V>`) - a homogeneous slice (and mutable slice) of `Copy` types generic over the virtual function table, 
 - `SliceDrop<V>` (and `SliceDropMut<V>`) - a homogeneous slice (and mutable slice) of types generic over the virtual function table, 
 - `BoxValue<V>` - an untyped boxed value of any size.
 - `SmallValue<V>` - an untyped value that fits into a `usize`.
 - `ValueRef<V>` (and `ValueMut<V>`) - an untyped value reference (and mutable reference).
 - `CopyValueRef<V>` (and `CopyValueMut<V>`) - an untyped value reference (and mutable reference) of
   a `Copy` type.

The main difference between the `Copy` variants of these is that they do not require a designated drop
function pointer, which makes them simpler and potentially more performant. However, the benchmarks
have not revealed any performance advantages in the `Copy` variants, so it is recommended to always
use the `Drop` variants by default. Furthermore the `Copy` variants may be deprecated in the future to
simplify the API.


# Examples

Create homogeneous untyped `Vec`s that store a single virtual function table for all contained
elements:
```rust
use dync::VecDrop;
// Create an untyped `Vec`.
let vec: VecDrop = vec![1_i32,2,3,4].into();
// Access elements either by downcasting to the underlying type.
for value_ref in vec.iter() {
    let int = value_ref.downcast::<i32>().unwrap();
    println!("{}", int);
}
// Or downcast the iterator directly for more efficient traversal.
for int in vec.iter_as::<i32>().unwrap() {
    println!("{}", int);
}
```

The `VecDrop` type above defaults to the empty virtual table (with the exception of the drop
function), which is not terribly useful when the contained values need to be processed in
some way.  `dync` provides support for common standard library traits such as:
- `Drop`
- `Clone`
- `PartialEq`
- `std::hash::Hash`
- `std::fmt::Debug`
- `Send` and `Sync`
- more to come

So to produce a `VecDrop` of a printable type, we could instead do
```rust
use dync::{VecDrop, traits::DebugVTable};
// Create an untyped `Vec` of `std::fmt::Debug` types.
let vec: VecDrop<DebugVTable> = vec![1_i32,2,3,4].into();
// We can now iterate and print value references (which inherit the VTable from the container)
// without needing a downcast.
for value_ref in vec.iter() {
    println!("{:?}", value_ref);
}
```

See the [`exmaples`](/examples) directory for more.

# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
