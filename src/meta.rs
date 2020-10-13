use crate::copy_value::*;
use crate::elem::*;
#[cfg(feature = "traits")]
use crate::slice::*;
use crate::slice_copy::*;
#[cfg(feature = "traits")]
use crate::value::*;
use crate::vec_copy::*;
use crate::vtable::*;

#[cfg(feature = "traits")]
use crate::traits::*;

/// Defines a meta struct containing information about a type but not the type itself.
///
/// Note that here V would typically represent a pointer type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta<V> {
    pub elem: ElemInfo,
    pub vtable: V,
}

#[cfg(feature = "traits")]
impl<'a, V: HasDrop> From<ValueRef<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(val: ValueRef<'a, V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            elem: ElemInfo {
                type_id: val.type_id,
                size: val.bytes.len() / val.alignment,
                alignment: val.alignment,
            },
            vtable: val.vtable,
        }
    }
}

#[cfg(feature = "traits")]
impl<'a, V: HasDrop> From<ValueMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(val: ValueMut<'a, V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            elem: ElemInfo {
                type_id: val.type_id,
                size: val.bytes.len() / val.alignment,
                alignment: val.alignment,
            },
            vtable: val.vtable,
        }
    }
}

impl<'a, V> From<CopyValueRef<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(val: CopyValueRef<'a, V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            elem: ElemInfo {
                type_id: val.type_id,
                size: val.bytes.len() / val.alignment,
                alignment: val.alignment,
            },
            vtable: val.vtable,
        }
    }
}

impl<'a, V> From<SliceCopy<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceCopy<'a, V>) -> Self {
        Meta {
            elem: slice.elem,
            vtable: slice.vtable,
        }
    }
}

impl<'a, V> From<SliceCopyMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceCopyMut<'a, V>) -> Self {
        Meta {
            elem: slice.elem,
            vtable: slice.vtable,
        }
    }
}

#[cfg(feature = "traits")]
impl<'a, V> From<Slice<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: Slice<'a, V>) -> Self {
        Meta::from(slice.data)
    }
}

#[cfg(feature = "traits")]
impl<'a, V> From<SliceMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceMut<'a, V>) -> Self {
        Meta::from(slice.data)
    }
}

impl<'a, V> From<&'a VecCopy<V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(vec: &'a VecCopy<V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            elem: vec.data.elem,
            vtable: VTableRef::Ref(vec.vtable.as_ref()),
        }
    }
}
