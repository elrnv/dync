use std::any::TypeId;

// At the time of this writing, there is no evidence that there is a significant benefit in sharing
// vtables via Rc or Arc, but to make potential future refactoring easier we use the Ptr alias.
use std::boxed::Box as Ptr;

use crate::copy_value::*;
use crate::slice_copy::*;
use crate::slice_drop::*;
use crate::traits::*;
use crate::value::*;
use crate::vec_copy::*;
use crate::vtable::*;

/// Defines a meta struct containing information about a type but not the type itself.
///
/// Note that here V would typically represent a pointer type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta<V> {
    pub element_size: usize,
    pub element_type_id: TypeId,
    pub vtable: V,
}

impl<'a, B: GetBytesMut, V: Clone + HasDrop> From<&'a Value<B, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(val: &'a Value<B, V>) -> Meta<Ptr<V>> {
        Meta {
            element_size: val.bytes.get_bytes_ref().len(),
            element_type_id: val.type_id,
            vtable: Ptr::clone(&val.vtable),
        }
    }
}
impl<'a, V: HasDrop> From<ValueRef<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(val: ValueRef<'a, V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            element_size: val.bytes.len(),
            element_type_id: val.type_id,
            vtable: val.vtable,
        }
    }
}

impl<'a, V: HasDrop> From<ValueMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(val: ValueMut<'a, V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            element_size: val.bytes.len(),
            element_type_id: val.type_id,
            vtable: val.vtable,
        }
    }
}

impl<'a, V> From<SliceCopy<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceCopy<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable,
        }
    }
}

impl<'a, V: Clone> From<SliceCopy<'a, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(slice: SliceCopy<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable.into_owned(),
        }
    }
}

impl<'a, V> From<SliceCopyMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceCopyMut<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable,
        }
    }
}

impl<'a, V: Clone> From<SliceCopyMut<'a, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(slice: SliceCopyMut<'a, V>) -> Self {
        Meta {
            element_size: slice.element_size,
            element_type_id: slice.element_type_id,
            vtable: slice.vtable.into_owned(),
        }
    }
}

impl<'a, V> From<SliceDrop<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceDrop<'a, V>) -> Self {
        Meta::from(slice.data)
    }
}

impl<'a, V: Clone> From<SliceDrop<'a, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(slice: SliceDrop<'a, V>) -> Self {
        Meta::from(slice.data)
    }
}

impl<'a, V> From<SliceDropMut<'a, V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(slice: SliceDropMut<'a, V>) -> Self {
        Meta::from(slice.data)
    }
}

impl<'a, V: Clone> From<SliceDropMut<'a, V>> for Meta<Ptr<V>> {
    #[inline]
    fn from(slice: SliceDropMut<'a, V>) -> Self {
        Meta::from(slice.data)
    }
}

impl<'a, V> From<&'a VecCopy<V>> for Meta<VTableRef<'a, V>> {
    #[inline]
    fn from(vec: &'a VecCopy<V>) -> Meta<VTableRef<'a, V>> {
        Meta {
            element_size: vec.element_size,
            element_type_id: vec.element_type_id,
            vtable: VTableRef::Ref(vec.vtable.as_ref()),
        }
    }
}

impl<'a, V> From<&'a VecCopy<V>> for Meta<Ptr<V>>
where
    Ptr<V>: Clone,
{
    #[inline]
    fn from(vec: &'a VecCopy<V>) -> Meta<Ptr<V>> {
        Meta {
            element_size: vec.element_size,
            element_type_id: vec.element_type_id,
            vtable: vec.vtable.clone(),
        }
    }
}
