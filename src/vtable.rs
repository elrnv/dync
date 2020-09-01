// At the time of this writing, there is no evidence that there is a significant benefit in sharing
// vtables via Rc or Arc, but to make potential future refactoring easier we use the Ptr alias.
use std::boxed::Box as Ptr;

/// `VTable` defines a type that represents a virtual function table for some type `T`.
///
/// `T` is different than a type that can be turned into a trait object like `Box<dyn Any>`
/// because a `VTable` effectively decouples the type's behaviour from the data it contains.
///
/// This mechanism allows the virtual function table to be attached to a homogeneous container, to
/// prevent storing duplicates of these tables for each type instance stored in the container.
///
/// This is precisely how it is used to build `VecDyn<V>`, which is generic over the virtual table
/// rather than the type itself.
pub trait VTable<T> {
    fn build_vtable() -> Self;
}

impl<T: Copy> VTable<T> for () {
    #[inline]
    fn build_vtable() -> Self {}
}

#[cfg(feature = "traits")]
impl<T: crate::traits::DropBytes, V: VTable<T>> VTable<T> for (crate::traits::DropFn, V) {
    #[inline]
    fn build_vtable() -> Self {
        (T::drop_bytes, V::build_vtable())
    }
}

/// A VTable reference type.
///
/// Note we always need Drop because it's possible to clone ValueRef's contents, which need to know
/// how to drop themselves.
#[derive(Clone, Debug)]
pub enum VTableRef<'a, V>
where
    V: ?Sized,
{
    Ref(&'a V),
    Box(Box<V>),
    #[cfg(feature = "shared-vtables")]
    Rc(Rc<V>),
}

impl<'a, V: Clone + ?Sized> VTableRef<'a, V> {
    #[inline]
    pub fn take(self) -> V {
        match self {
            VTableRef::Ref(v) => v.clone(),
            VTableRef::Box(v) => *v,
            #[cfg(feature = "shared-vtables")]
            VTableRef::Rc(v) => Rc::try_unwrap(v).unwrap_or_else(|v| (*v).clone()),
        }
    }

    #[inline]
    pub fn into_owned(self) -> Ptr<V> {
        match self {
            VTableRef::Ref(v) => Ptr::new(v.clone()),
            VTableRef::Box(v) => v,
            #[cfg(feature = "shared-vtables")]
            VTableRef::Rc(v) => Rc::clone(&v),
        }
    }
}

impl<'a, V: ?Sized> std::ops::Deref for VTableRef<'a, V> {
    type Target = V;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, V: ?Sized> From<&'a V> for VTableRef<'a, V> {
    #[inline]
    fn from(v: &'a V) -> VTableRef<'a, V> {
        VTableRef::Ref(v)
    }
}

impl<'a, V: ?Sized> From<Box<V>> for VTableRef<'a, V> {
    #[inline]
    fn from(v: Box<V>) -> VTableRef<'a, V> {
        VTableRef::Box(v)
    }
}

#[cfg(feature = "shared-vtables")]
impl<'a, V: ?Sized> From<Ptr<V>> for VTableRef<'a, V> {
    #[inline]
    fn from(v: Ptr<V>) -> VTableRef<'a, V> {
        VTableRef::Rc(v)
    }
}

impl<'a, V: ?Sized> AsRef<V> for VTableRef<'a, V> {
    #[inline]
    fn as_ref(&self) -> &V {
        match self {
            VTableRef::Ref(v) => v,
            VTableRef::Box(v) => &*v,
            #[cfg(feature = "shared-vtables")]
            VTableRef::Rc(v) => &*v,
        }
    }
}
