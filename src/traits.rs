//! This module defines the function pointers for supported traits from the standard library.
//!
//! `CloneFromFn` and `DropFn` enable the use of `VecClone`.
//!
//! The remaining traits improve compatibility with the rest of the standard library.

use crate::bytes::*;
use crate::vtable::VTable;
use dync_derive::dync_trait_method;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::{ManuallyDrop, MaybeUninit};

pub trait DropBytes {
    #[doc(hidden)]
    unsafe fn drop_bytes(bytes: &mut [MaybeUninit<u8>]);
}

pub trait CloneBytes: Clone {
    #[dync_trait_method]
    fn clone(&self) -> Self;
    //unsafe fn clone_bytes(src: &[MaybeUninit<u8>]) -> Box<[MaybeUninit<u8>]>;
    #[dync_trait_method]
    fn clone_from(&mut self, src: &Self);
    //unsafe fn clone_from_bytes(dst: &mut [MaybeUninit<u8>], src: &[MaybeUninit<u8>]);

    #[doc(hidden)]
    unsafe fn clone_into_raw_bytes(src: &[MaybeUninit<u8>], dst: &mut [MaybeUninit<u8>]);
}

pub trait PartialEqBytes: PartialEq {
    #[dync_trait_method]
    fn eq(&self, other: &Self) -> bool;
    //unsafe fn eq_bytes(a: &[MaybeUninit<u8>], b: &[MaybeUninit<u8>]) -> bool;
}

pub trait HashBytes: Hash {
    #[dync_trait_method]
    fn hash<H: Hasher>(&self, state: &mut H);
    //unsafe fn hash_bytes(bytes: &[MaybeUninit<u8>], state: &mut dyn Hasher);
}

pub trait DebugBytes: fmt::Debug {
    #[dync_trait_method]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>;
    //unsafe fn fmt_bytes(bytes: &[MaybeUninit<u8>], f: &mut fmt::Formatter) -> Result<(), fmt::Error>;
}

impl<T: 'static> DropBytes for T {
    #[inline]
    unsafe fn drop_bytes(bytes: &mut [MaybeUninit<u8>]) {
        let md: &mut ManuallyDrop<T> = Bytes::from_bytes_mut(bytes);
        ManuallyDrop::drop(md);
    }
}

impl<T: Clone + 'static> CloneBytes for T {
    #[inline]
    unsafe fn clone_bytes(src: &[MaybeUninit<u8>]) -> Box<[MaybeUninit<u8>]> {
        let typed_src: &T = Bytes::from_bytes(src);
        Bytes::box_into_box_bytes(Box::new(typed_src.clone()))
    }
    #[inline]
    unsafe fn clone_from_bytes(dst: &mut [MaybeUninit<u8>], src: &[MaybeUninit<u8>]) {
        let typed_src: &T = Bytes::from_bytes(src);
        let typed_dst: &mut T = Bytes::from_bytes_mut(dst);
        typed_dst.clone_from(typed_src);
    }
    #[inline]
    unsafe fn clone_into_raw_bytes(src: &[MaybeUninit<u8>], dst: &mut [MaybeUninit<u8>]) {
        let typed_src: &T = Bytes::from_bytes(src);
        let cloned = T::clone(typed_src);
        let cloned_bytes = Bytes::as_bytes(&cloned);
        dst.copy_from_slice(cloned_bytes);
        let _ = ManuallyDrop::new(cloned);
    }
}

impl<T: PartialEq + 'static> PartialEqBytes for T {
    #[inline]
    unsafe fn eq_bytes(a: &[MaybeUninit<u8>], b: &[MaybeUninit<u8>]) -> bool {
        let (a, b): (&T, &T) = (Bytes::from_bytes(a), Bytes::from_bytes(b));
        a.eq(b)
    }
}

impl<T: Hash + 'static> HashBytes for T {
    #[inline]
    unsafe fn hash_bytes(bytes: &[MaybeUninit<u8>], mut state: &mut dyn Hasher) {
        let typed_data: &T = Bytes::from_bytes(bytes);
        typed_data.hash(&mut state)
    }
}

impl<T: fmt::Debug + 'static> DebugBytes for T {
    #[inline]
    unsafe fn fmt_bytes(
        bytes: &[MaybeUninit<u8>],
        f: &mut fmt::Formatter,
    ) -> Result<(), fmt::Error> {
        let typed_data: &T = Bytes::from_bytes(bytes);
        typed_data.fmt(f)
    }
}

pub type CloneFn = unsafe fn(&[MaybeUninit<u8>]) -> Box<[MaybeUninit<u8>]>;
pub type CloneFromFn = unsafe fn(&mut [MaybeUninit<u8>], &[MaybeUninit<u8>]);
pub type CloneIntoRawFn = unsafe fn(&[MaybeUninit<u8>], &mut [MaybeUninit<u8>]);
pub type EqFn = unsafe fn(&[MaybeUninit<u8>], &[MaybeUninit<u8>]) -> bool;
pub type HashFn = unsafe fn(&[MaybeUninit<u8>], &mut dyn Hasher);
pub type FmtFn = unsafe fn(&[MaybeUninit<u8>], &mut fmt::Formatter) -> Result<(), fmt::Error>;
pub type DropFn = unsafe fn(&mut [MaybeUninit<u8>]);

use downcast_rs::{impl_downcast, Downcast};

pub trait HasDrop: Downcast {
    fn drop_fn(&self) -> &DropFn;
}

impl_downcast!(HasDrop);

pub trait HasClone {
    fn clone_fn(&self) -> &CloneFn;
    fn clone_from_fn(&self) -> &CloneFromFn;
    fn clone_into_raw_fn(&self) -> &CloneIntoRawFn;
}

pub trait HasHash {
    fn hash_fn(&self) -> &HashFn;
}

pub trait HasPartialEq {
    fn eq_fn(&self) -> &EqFn;
}

pub trait HasEq: HasPartialEq {}

pub trait HasDebug {
    fn fmt_fn(&self) -> &FmtFn;
}

/// # Safety
///
/// Implementing containers must contain types that implement `Send`.
pub unsafe trait HasSend {}

/// # Safety
///
/// Implementing containers must contain types that implement `Sync`.
pub unsafe trait HasSync {}

#[derive(Clone)]
pub struct DropVTable(pub DropFn);
#[derive(Clone)]
pub struct CloneVTable(pub DropFn, pub CloneFn, pub CloneFromFn, pub CloneIntoRawFn);
#[derive(Clone)]
pub struct PartialEqVTable(pub DropFn, pub EqFn);
#[derive(Clone)]
pub struct EqVTable(pub DropFn, pub EqFn);
#[derive(Clone)]
pub struct HashVTable(pub DropFn, pub HashFn);
#[derive(Clone)]
pub struct DebugVTable(pub DropFn, pub FmtFn);
#[derive(Clone)]
pub struct SendVTable(pub DropFn);
#[derive(Clone)]
pub struct SyncVTable(pub DropFn);

// VTable implementations are needed for builtin types to work with builtin vtables
impl<T: DropBytes> VTable<T> for DropVTable {
    fn build_vtable() -> Self {
        DropVTable(T::drop_bytes)
    }
}

impl<T: DropBytes + CloneBytes> VTable<T> for CloneVTable {
    fn build_vtable() -> Self {
        CloneVTable(
            T::drop_bytes,
            T::clone_bytes,
            T::clone_from_bytes,
            T::clone_into_raw_bytes,
        )
    }
}

impl<T: DropBytes + PartialEqBytes> VTable<T> for PartialEqVTable {
    fn build_vtable() -> Self {
        PartialEqVTable(T::drop_bytes, T::eq_bytes)
    }
}

impl<T: DropBytes + PartialEqBytes> VTable<T> for EqVTable {
    fn build_vtable() -> Self {
        EqVTable(T::drop_bytes, T::eq_bytes)
    }
}

impl<T: DropBytes + HashBytes> VTable<T> for HashVTable {
    fn build_vtable() -> Self {
        HashVTable(T::drop_bytes, T::hash_bytes)
    }
}

impl<T: DropBytes> VTable<T> for SendVTable {
    fn build_vtable() -> Self {
        SendVTable(T::drop_bytes)
    }
}

impl<T: DropBytes> VTable<T> for SyncVTable {
    fn build_vtable() -> Self {
        SyncVTable(T::drop_bytes)
    }
}

impl<T: DropBytes + DebugBytes> VTable<T> for DebugVTable {
    fn build_vtable() -> Self {
        DebugVTable(T::drop_bytes, T::fmt_bytes)
    }
}

macro_rules! impl_has_drop {
    ($($trait:ident),*) => {
        $(
            impl HasDrop for $trait {
                fn drop_fn(&self) -> &DropFn {
                    &self.0
                }
            }
        )*
    }
}

impl_has_drop!(
    DropVTable,
    CloneVTable,
    PartialEqVTable,
    EqVTable,
    HashVTable,
    DebugVTable,
    SendVTable,
    SyncVTable
);

impl HasClone for CloneVTable {
    fn clone_fn(&self) -> &CloneFn {
        &self.1
    }
    fn clone_from_fn(&self) -> &CloneFromFn {
        &self.2
    }
    fn clone_into_raw_fn(&self) -> &CloneIntoRawFn {
        &self.3
    }
}

impl HasHash for HashVTable {
    fn hash_fn(&self) -> &HashFn {
        &self.1
    }
}

impl HasPartialEq for PartialEqVTable {
    fn eq_fn(&self) -> &EqFn {
        &self.1
    }
}

impl HasPartialEq for EqVTable {
    fn eq_fn(&self) -> &EqFn {
        &self.1
    }
}

impl HasEq for EqVTable {}

impl HasDebug for DebugVTable {
    fn fmt_fn(&self) -> &FmtFn {
        &self.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec_dyn::VecDyn;

    #[test]
    fn drop() {
        // Empty vec dropped
        let vd = VecDyn::<DropVTable>::with_type::<u32>();
        let v = vd.into_vec::<u32>().unwrap();
        assert_eq!(v, Vec::new());

        // Non-empty vec dropped
        let vd = VecDyn::<DropVTable>::from(vec![1u32, 2]);
        let v = vd.into_vec::<u32>().unwrap();
        assert_eq!(v, vec![1u32, 2]);
    }

    #[test]
    fn clone() {
        let v = VecDyn::<CloneVTable>::from(vec![1u32, 2]);
        let v_clone = v.clone();
        assert_eq!(
            v.into_vec::<u32>().unwrap(),
            v_clone.into_vec::<u32>().unwrap()
        );
    }

    #[test]
    fn partial_eq() {
        let a = VecDyn::<PartialEqVTable>::from(vec![1, 2]);
        let b = VecDyn::<PartialEqVTable>::from(vec![1, 2]);
        assert!(a == b);

        let a = VecDyn::<EqVTable>::from(vec![1, 2]);
        let b = VecDyn::<EqVTable>::from(vec![1, 2]);
        assert!(a == b);
    }

    #[test]
    fn hash() {
        use std::collections::hash_map::DefaultHasher;
        let a = VecDyn::<HashVTable>::from(vec![1, 2]);
        let b = VecDyn::<HashVTable>::from(vec![1, 2]);

        let mut s = DefaultHasher::new();
        a.hash(&mut s);
        let a_hash = s.finish();

        let mut s = DefaultHasher::new();
        b.hash(&mut s);
        let b_hash = s.finish();
        assert_eq!(a_hash, b_hash);
    }

    #[test]
    fn debug() {
        let a = VecDyn::<DebugVTable>::from(vec![1, 2]);
        eprintln!("{:?}", a);
    }
}
