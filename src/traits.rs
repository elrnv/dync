//! This module defines the function pointers for supported traits from the standard library.
//!
//! `CloneFromFn` and `DropFn` enable the use of `VecClone`.
//!
//! The remaining traits improve compatibility with the rest of the standard library.

use crate::bytes::*;
use dync_derive::dync_trait_method;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;

pub trait DropBytes {
    unsafe fn drop_bytes(bytes: &mut [u8]);
}

pub trait CloneBytes: Clone {
    #[dync_trait_method]
    fn clone(&self) -> Self;
    //unsafe fn clone_bytes(src: &[u8]) -> Box<[u8]>;
    #[dync_trait_method]
    fn clone_from(&mut self, src: &Self);
    //unsafe fn clone_from_bytes(dst: &mut [u8], src: &[u8]);

    /// Clone without dropping the destination bytes.
    unsafe fn clone_into_raw_bytes(src: &[u8], dst: &mut [u8]);
}

pub trait PartialEqBytes: PartialEq {
    #[dync_trait_method]
    fn eq(&self, other: &Self) -> bool;
    //unsafe fn eq_bytes(a: &[u8], b: &[u8]) -> bool;
}

pub trait HashBytes: Hash {
    #[dync_trait_method]
    fn hash<H: Hasher>(&self, state: &mut H);
    //unsafe fn hash_bytes(bytes: &[u8], state: &mut dyn Hasher);
}

pub trait DebugBytes: fmt::Debug {
    #[dync_trait_method]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>;
    //unsafe fn fmt_bytes(bytes: &[u8], f: &mut fmt::Formatter) -> Result<(), fmt::Error>;
}

impl<T: 'static> DropBytes for T {
    #[inline]
    unsafe fn drop_bytes(bytes: &mut [u8]) {
        let md: &mut ManuallyDrop<T> = Bytes::from_bytes_mut(bytes);
        ManuallyDrop::drop(md);
    }
}

impl<T: Clone + 'static> CloneBytes for T {
    #[inline]
    unsafe fn clone_bytes(src: &[u8]) -> Box<[u8]> {
        let typed_src: &T = Bytes::from_bytes(src);
        Bytes::box_into_box_bytes(Box::new(typed_src.clone()))
    }
    #[inline]
    unsafe fn clone_from_bytes(dst: &mut [u8], src: &[u8]) {
        let typed_src: &T = Bytes::from_bytes(src);
        let typed_dst: &mut T = Bytes::from_bytes_mut(dst);
        typed_dst.clone_from(typed_src);
    }
    #[inline]
    unsafe fn clone_into_raw_bytes(src: &[u8], dst: &mut [u8]) {
        let typed_src: &T = Bytes::from_bytes(src);
        let cloned = T::clone(typed_src);
        let cloned_bytes = Bytes::as_bytes(&cloned);
        dst.copy_from_slice(cloned_bytes);
        let _ = ManuallyDrop::new(cloned);
    }
}

impl<T: PartialEq + 'static> PartialEqBytes for T {
    #[inline]
    unsafe fn eq_bytes(a: &[u8], b: &[u8]) -> bool {
        let (a, b): (&T, &T) = (Bytes::from_bytes(a), Bytes::from_bytes(b));
        a.eq(b)
    }
}

impl<T: Hash + 'static> HashBytes for T {
    #[inline]
    unsafe fn hash_bytes(bytes: &[u8], mut state: &mut dyn Hasher) {
        let typed_data: &T = Bytes::from_bytes(bytes);
        typed_data.hash(&mut state)
    }
}

impl<T: fmt::Debug + 'static> DebugBytes for T {
    #[inline]
    unsafe fn fmt_bytes(bytes: &[u8], f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let typed_data: &T = Bytes::from_bytes(bytes);
        typed_data.fmt(f)
    }
}

pub type CloneFn = unsafe fn(&[u8]) -> Box<[u8]>;
pub type CloneFromFn = unsafe fn(&mut [u8], &[u8]);
pub type CloneIntoRawFn = unsafe fn(&[u8], &mut [u8]);
pub type EqFn = unsafe fn(&[u8], &[u8]) -> bool;
pub type HashFn = unsafe fn(&[u8], &mut dyn Hasher);
pub type FmtFn = unsafe fn(&[u8], &mut fmt::Formatter) -> Result<(), fmt::Error>;
pub(crate) type DropFn = unsafe fn(&mut [u8]);

pub(crate) trait HasDrop {
    fn drop_fn(&self) -> &DropFn;
}

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

pub struct CloneVTable(pub CloneFn, pub CloneFromFn, pub CloneIntoRawFn);
pub struct DropVTable(pub DropFn);
pub struct PartialEqVTable(pub EqFn);
pub struct EqVTable(pub EqFn);
pub struct HashVTable(pub HashFn);
pub struct DebugVTable(pub FmtFn);

impl HasDrop for DropVTable {
    fn drop_fn(&self) -> &DropFn {
        &self.0
    }
}

impl HasClone for CloneVTable {
    fn clone_fn(&self) -> &CloneFn {
        &self.0
    }
    fn clone_from_fn(&self) -> &CloneFromFn {
        &self.1
    }
    fn clone_into_raw_fn(&self) -> &CloneIntoRawFn {
        &self.2
    }
}

impl HasHash for HashVTable {
    fn hash_fn(&self) -> &HashFn {
        &self.0
    }
}

impl HasPartialEq for PartialEqVTable {
    fn eq_fn(&self) -> &EqFn {
        &self.0
    }
}

impl HasPartialEq for EqVTable {
    fn eq_fn(&self) -> &EqFn {
        &self.0
    }
}

impl HasEq for EqVTable {}

impl HasDebug for DebugVTable {
    fn fmt_fn(&self) -> &FmtFn {
        &self.0
    }
}
